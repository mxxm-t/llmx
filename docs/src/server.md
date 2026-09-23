# `src/server/` - the multi-user server

`llmx serve`, in three headers above the inference layer, designed in
`docs/SERVER.md`. No external libraries: sockets, HTTP/1.1, JSON and the
scheduler are the runtime's own.

- `http.hpp`: `Listener`, `Connection`, `Request`. Blocking BSD or Winsock
  sockets, one thread per connection; a request is read with size limits
  (413 and 431 past them, 400 for a malformed line), a reply is written
  whole or as a chunked stream, and a client's disconnect surfaces as a
  failed write rather than a signal. The `http` CTest drives it with its
  own client.
- `scheduler.hpp`: `Request` and `Scheduler`. Connection threads submit
  requests and drain their token channels; the scheduler thread is the
  single caller of `Model::forward`. The KV pool holds `--ctx-size` tokens
  in total; past `--max-queue` waiting requests a new one is refused with
  503. Each iteration admits queued requests the pool can hold (a capped
  request's prompt plus `max_tokens`, an uncapped one's prompt plus a
  step it grows by as it generates, pausing the latest admitted uncapped
  request when the pool runs out; a paused request's history stays as a
  donor and it resumes from it), assembles one pass of every
  decoding request's next token plus prompt slices up to `ubatch`, each
  slice carrying its whole prompt's extent so it takes the kernels one pass
  over the prompt would, samples per request with its own seeded state,
  and finishes on EOS, a stop string or the token limit. A finished
  request's history stays as a donor: a later prompt repeating its tokens
  forks the shared full blocks and prefills only the rest. Cancellation is
  a flag seen at the next iteration; blocks return once the device has
  retired the pass that read them.
- `api.hpp`: the routes and `serve()`. Native `/v1/generate`, `/v1/chat`
  and `/v1/health`; the OpenAI-compatible `/v1/chat/completions`,
  `/v1/completions` and `/v1/models`, one parse, one request and one drain
  loop shared with the native routes, with the clients' synonyms accepted
  and errors in their shape. On the compatible routes an absent
  `max_tokens`, or -1, means no cap: the reply may run to the end of the
  request's context; the native routes keep a default of 64. The
  compatible replies carry a `timings` object beside `usage`, and every
  finished request logs a line on stderr. Streams hold a character split across tokens
  until it completes and replace invalid UTF-8 with U+FFFD.
