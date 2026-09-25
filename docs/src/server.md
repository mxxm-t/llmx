# `src/server/` - the multi-user server

`llmx serve`, in three headers above the inference layer, designed in
`docs/SERVER.md`. No external libraries: sockets, HTTP/1.1, JSON and the
scheduler are the runtime's own.

- `http.hpp`: `Listener`, `Connection`, `Request`.
  Blocking BSD or Winsock sockets, one thread per connection; a request is read with size limits (413 and 431 past them, 400 for a malformed line), a reply is written whole or as a chunked stream, and a client's disconnect surfaces as a failed write, which throws `ClientGone`, rather than a signal.
  `peer_closed` asks without writing, blocking or taking a byte whether the client has closed the connection: `poll` or `WSAPoll` with no wait, then a one-byte `MSG_PEEK` (with `MSG_DONTWAIT` where the platform has it) whose end of stream or reset means closed and whose data means open, so a client that shuts only its sending side reads as closed.
  On Windows the poll asks for normal data only (`POLLRDNORM`), since its `POLLIN` also takes urgent (out-of-band) data, and the peek there, which has no `MSG_DONTWAIT`, would then wait for normal data that may never come.
  `Listener::accept` returns no connection only once `close()` has run, and any other failure is retried: at once when it is one client's (a connection aborted before it was taken, an interrupted call), and after 50 ms otherwise, so a client's aborted connection never stops the server and an error that persists cannot spin a core.
  `respond` throws while a stream is open, since a second head would land inside the chunked body.
  The `http` CTest drives it with its own client.
- `scheduler.hpp`: `Request` and `Scheduler`.
  Connection threads submit requests and drain their token channels; the scheduler thread is the single caller of `Model::forward`.
  `Request::next` waits for a token or the end only until a deadline, so a connection thread can look at its client between tokens.
  The KV pool holds `--ctx-size` tokens in total, and `submit` owns what one request may hold (`token_limit`): it refuses a prompt plus `max_tokens` past it, or an uncapped prompt that fills it, and sets an uncapped request's `max_tokens` to the room its prompt leaves; past `--max-queue` waiting requests a new one is refused with 503.
  Each iteration admits queued requests the pool can hold (a capped request's prompt plus `max_tokens`, an uncapped one's prompt plus a step it grows by as it generates, pausing the latest admitted uncapped request when the pool runs out; a paused request's history stays as a donor and it resumes from it), assembles one pass of every decoding request's next token plus prompt slices up to the model's prompt batch (`Model::prefill_batch`), each slice carrying its whole prompt's extent so it takes the kernels one pass over the prompt would, samples per request with its own seeded state, and finishes on EOS, a stop string or the token limit.
  A finished request's history stays as a donor: a later prompt repeating its tokens forks the shared full blocks and prefills only the rest.
  The donor a request forks is chosen before room is made and the other donors go first; if the pool is still short the chosen one is consumed: only the shared blocks pass to the request and the rest are freed, so shared blocks are reserved once and a follow-up turn keeps the history it repeats.
  A request sharing every full block of its donor, a follow-up turn or a resume, consumes it before any other donor goes, so an unrelated donor stays whenever consuming that one makes room.
  Cancellation is a flag seen at the next iteration, which ends a queued request wherever it waits and drops an active one from the batch; blocks return once the device has retired the pass that read them.
- `api.hpp`: the routes and `serve()`.
  Native `/v1/generate`, `/v1/chat` and `/v1/health`; the OpenAI-compatible `/v1/chat/completions`, `/v1/completions` and `/v1/models`, one parse, one request and one drain loop shared with the native routes, with the clients' synonyms accepted and errors in their shape.
  The path of a generating route resolves to its `Route` once, and `compat(Route)` alone says which of those routes speak the clients' shape, so a compatible request refused while it is read gets that shape too.
  `integer` refuses a whole-number field that is fractional or outside what it is cast to (`max_tokens`, `max_completion_tokens`, `top_k` and `n` within `int`, `seed` from 0 to below 2^64) with 400 before any cast.
  On the compatible routes an absent `max_tokens`, or -1, means no cap (`until_limit`): the reply may run to the end of the request's context; the native routes keep a default of 64 and refuse -1 as any other cap below 1.
  A `seed` of -1, which clients send for a random one, is no seed on the compatible routes, sampled as a request with none is, and the native routes refuse it as any other seed below 0.
  The routes only map the scheduler's refusals to statuses.
  The drain loop cancels a request when a write to its client fails, and between writes looks at the socket every 100 ms (`kProbe`), token or not, cancelling once `peer_closed` says the client has gone, so a client that leaves while its request is queued, prefilling or building a whole reply is noticed within about 100 ms as well.
  Either way a `ClientGone` ends the loop, and `handle` answers it with nothing, not even a 500, so the connection just closes.
  The compatible replies carry a `timings` object beside `usage`, and every finished request logs a line on stderr; both take the decode rate from `Request::Timings`.
  A stream whose pass fails ends with one `data:` event holding the error in the route's shape, and no `data: [DONE]`, since its 200 head has gone out.
  Streams hold a character split across tokens until it completes and replace invalid UTF-8 with U+FFFD.
