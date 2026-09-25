# `src/server/` - the multi-user server

`llmx serve`, in three headers above the inference layer, designed in
`docs/SERVER.md`. No external libraries: sockets, HTTP/1.1, JSON and the
scheduler are the runtime's own.

- `http.hpp`: `Listener`, `Connection`, `Request`.
  Blocking BSD or Winsock sockets, one thread per connection; a request is read with size limits (413 and 431 past them, 400 for a malformed line), a reply is written whole or as a chunked stream, and a client's disconnect surfaces as a failed write rather than a signal.
  `Listener::accept` returns no connection only once `close()` has run, and any other failure is retried: at once when it is one client's (a connection aborted before it was taken, an interrupted call), and after 50 ms otherwise, so a client's aborted connection never stops the server and an error that persists cannot spin a core.
  `respond` throws while a stream is open, since a second head would land inside the chunked body.
  The `http` CTest drives it with its own client.
- `scheduler.hpp`: `Request` and `Scheduler`.
  Connection threads submit requests and drain their token channels; the scheduler thread is the single caller of `Model::forward`.
  The KV pool holds `--ctx-size` tokens in total, and `submit` owns what one request may hold (`token_limit`): it refuses a prompt plus `max_tokens` past it, or an uncapped prompt that fills it, and sets an uncapped request's `max_tokens` to the room its prompt leaves; past `--max-queue` waiting requests a new one is refused with 503.
  Each iteration admits queued requests the pool can hold (a capped request's prompt plus `max_tokens`, an uncapped one's prompt plus a step it grows by as it generates, pausing the latest admitted uncapped request when the pool runs out; a paused request's history stays as a donor and it resumes from it), assembles one pass of every decoding request's next token plus prompt slices up to the model's prompt batch (`Model::prefill_batch`), each slice carrying its whole prompt's extent so it takes the kernels one pass over the prompt would, samples per request with its own seeded state, and finishes on EOS, a stop string or the token limit.
  A finished request's history stays as a donor: a later prompt repeating its tokens forks the shared full blocks and prefills only the rest.
  Cancellation is a flag seen at the next iteration; blocks return once the device has retired the pass that read them.
- `api.hpp`: the routes and `serve()`.
  Native `/v1/generate`, `/v1/chat` and `/v1/health`; the OpenAI-compatible `/v1/chat/completions`, `/v1/completions` and `/v1/models`, one parse, one request and one drain loop shared with the native routes, with the clients' synonyms accepted and errors in their shape.
  The path of a generating route resolves to its `Route` once, and `compat(Route)` alone says which of those routes speak the clients' shape, so a compatible request refused while it is read gets that shape too.
  `integer` refuses a whole-number field that is fractional or outside what it is cast to (`max_tokens`, `max_completion_tokens`, `top_k` and `n` within `int`, `seed` from 0 to below 2^64) with 400 before any cast.
  On the compatible routes an absent `max_tokens`, or -1, means no cap (`until_limit`): the reply may run to the end of the request's context; the native routes keep a default of 64 and refuse -1 as any other cap below 1.
  A `seed` of -1, which clients send for a random one, is no seed on the compatible routes, sampled as a request with none is, and the native routes refuse it as any other seed below 0.
  The routes only map the scheduler's refusals to statuses.
  The compatible replies carry a `timings` object beside `usage`, and every finished request logs a line on stderr; both take the decode rate from `Request::Timings`.
  A stream whose pass fails ends with one `data:` event holding the error in the route's shape, and no `data: [DONE]`, since its 200 head has gone out.
  Streams hold a character split across tokens until it completes and replace invalid UTF-8 with U+FFFD.
