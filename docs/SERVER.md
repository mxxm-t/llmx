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
  blocks read-only through `Model::fork` and prefills only what follows.
  Finished requests stay a while as donors; the donor with the longest
  run of matching full blocks is found by comparing tokens, not hashes,
  since a server holds at most `max_seqs` donors for one model
  ([KV-CACHE](KV-CACHE.md), "Prefix sharing"). A live request's growing
  history is never shared.
- **A memory budget that admits, not crashes.** The KV pool has
  `max_blocks`. A request is admitted when the pool can hold its prompt and
  its `max_tokens`; otherwise it waits in the queue. An admitted request is
  never evicted; donors are, oldest first, when a request needs their
  blocks. A request that cannot be admitted is not started.
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
                 /v1/health, /v1/models
cli/main.cpp     `llmx serve <model.gguf> [--host H] [--port N] [--device D]
                 [--max-seqs N] [--ubatch N] [--cache-type-k T]
                 [--cache-type-v T] [--threads N]`
```

`server/` sits above `inference/` in the layering: it uses the model, the
tokenizer, the sampler and the chat template renderer and adds only
scheduling and transport. The directory is created with its first working
route, not before.

### Threads

```
accept thread ---> connection thread (one per socket)
                     parse request, validate, tokenize, render the chat
                     template, enqueue a Request, then block on the
                     request's token channel and write chunks until done
scheduler thread   the only caller of Model::forward for its devices
```

A `Request` carries the prompt ids, the sampling parameters, a `Sequence`,
the stop conditions and a channel: a mutex, a condition variable and a
deque of sampled ids that the connection thread drains. Cancellation is a
flag the connection thread sets when the socket closes; the scheduler sees
it at the next iteration, drops the request from the batch and releases
its sequence, which returns its blocks once the last pass that read them
has retired.

### The scheduler loop

```
loop:
  admit:   while the queue has a request and the pool can hold its prompt
           plus max_tokens (dropping donors, oldest first, to make room),
           and active < max_seqs: take it, find the donor sharing the
           longest run of full blocks, fork it and roll the fork back to
           the shared blocks, or make a fresh sequence; mark the request
           "prefilling" with an offset into its prompt
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

Two execution contexts are the planned overlap: the host samples one pass
while the device runs the next. On this runtime the host work per step is
small against a device pass, so the second context is added only when a
measurement shows the gap; the first version runs one context.

### Sampling

Sampling is per request, on the host, from the logits row the pass returns
for that entry: the existing `inference/sampler.hpp` with the request's own
temperature, top-k, top-p, penalty and seeded RNG, so a request with
`seed` set is reproducible regardless of what it was batched with. Greedy
requests give the text the CLI gives for the same prompt, which is the
first correctness gate below.

### Protocol

```
POST /v1/generate    {"prompt": "...", "max_tokens": 64, "temperature": 0.8,
                      "top_k": 40, "top_p": 0.95, "seed": 0, "stop": ["..."],
                      "stream": true}
POST /v1/chat        {"messages": [{"role": "user", "content": "..."}], ...}
                     the model's chat template renders the prompt
GET  /v1/health      {"status": "ok", "model": "...", "active": n, "queued": m,
                      "donors": d, "prefix_hits": h, "prefix_tokens": t}
GET  /v1/models      the loaded file, its quantization mix and context length
```

A streaming response is `text/event-stream`: one `data:` line per token
with the id and the decoded text, a final `data: [DONE]`, and the same
UTF-8 boundary rule the CLI streaming has, a split character is held until
its bytes complete. A non-streaming request gets one JSON object with the
text, the ids and the counts. Errors are JSON with an HTTP status: 400 for
a bad request, 413 for a prompt past the context, 503 when the queue is
full.

### What is not in the first version

- Eviction or preemption of admitted requests; admission is the budget.
- A prefix index that outlives the process, spans models or holds more
  than a few dozen donors: the hashed key of KV-CACHE is for that scale.
- Prefix reuse across two block sizes (a device and the CPU in one
  placement): the shared run is rounded to the first storage's block
  size, which every storage of one model shares today.
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
- **Throughput.** Aggregate decoded tokens per second at 1, 4, 8 and 16
  concurrent requests of the same shape, against the single-sequence
  `bench --model` figure at 1, on both backends; and against the reference
  runtime's server under the same load, same model, same card. A batch of
  one must not cost more than the CLI's decode; the gain at 4 and 8 is the
  reason the server exists and is reported, not assumed.
- **Long prompts under load.** A 16k prompt admitted while eight requests
  decode: the decoders' per-token latency during its prefill is recorded,
  since chunked prefill is what bounds it.

## Order of work

| # | Step | Test |
|---|---|---|
| 1 | `server/http.hpp`: listen, accept, parse a request, write a response and a chunked stream, on Windows and Linux (**done**) | The `http` CTest starts the server on a port, sends requests with the runtime's own client code and checks the responses, including a split chunk boundary |
| 2 | `server/scheduler.hpp`: queue, admission by pool budget, batch assembly with chunked prefill, sampling per request, channels, cancellation; `llmx serve` with `/v1/generate` and `/v1/health` (**done**; the throughput gate is open) | Greedy equality with the CLI alone and beside three decoders, cancellation, refusals (the `server` Python component, synthetic and real, CPU and device); the throughput gate at 1, 4, 8, 16 measured with `tools/server_load.py`: 119, 89, 81 and 103 percent of the reference's server on the device, the shortfall at 4 and 8 being the per-view dispatches of the device's batched attention path |
| 3 | `/v1/chat` through the template renderer; `/v1/models` (**done**) | A chat turn through the server in the `server` component |
| 3a | One dispatch per layer for `kv_write`, `attention` and `norm_rope_kv` over every view of a batch (**done**: a view table per dispatch, `shaders/views.glsl`) | The backend-vulkan checks over two views and the device HF gate; the throughput gate again: 118, 112, 109 and 125 percent of the reference's server at 1, 4, 8 and 16, met |
| 4 | Prefix reuse: finished requests kept as donors, the longest shared run of full blocks forked on admission (**done**) | The `server` component: a prompt repeating a 247-token excerpt with a different ending reuses the first request's blocks and its greedy text equals the CLI's; a 1995-token prefix on Qwen3-0.6B-Q8_0 costs 1.53 s the first time and 0.16 s with a donor on the device (8B: 8.74 s to 0.72 s; CPU 0.6B: 6.95 s to 0.95 s) |
| 5 | The second execution context, if measured to help | Throughput at 8 with and without it |
