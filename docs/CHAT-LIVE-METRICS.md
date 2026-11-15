# Chat live request metrics

Status: implementation and local qualification are complete. CPU byte identity,
companion/dashboard checks, real runtime-to-companion correlation, the MI50 server
component and hosted Chat on both operating systems pass. Runtime CI remains
pending. The optional MI50 real-model section skipped because no model fixture
was mounted. The feature is available on the isolated local preview at port 8185;
it is not merged or production-deployed. Owner: CHATUI. Base: `431dae029`.

## Contract

Show the current model call's phase, prompt progress and measured prompt and
generation rates in Chat using the existing Monitor collector. Text chunks are
not tokens. The API owns one public request key: an API-instance random nonce
and the scheduler's immutable admitted request number, in the opaque namespace
`cmpl-<nonce>-<number>`. Health rows and compatible response `id` fields use the
same value. Both chat and text completions use that namespace. No second
completion counter or correlation extension is needed, and callers must not
parse or join numeric suffixes.

For streaming compatible chat replies, emit the existing assistant-role chunk
immediately after successful admission and before waiting for a token. Its `id`
is the request key. Later chunks retain that same ID and content semantics,
without a second assistant-role opening. This adds one early transport write per
call, with no scheduler/device wait or change to per-token writes. The early
write makes the key observable during queueing and prefill. Requests refused
before admission have no key; admitted requests can later fail or be cancelled.

The companion validates a bounded opaque string and persists it as the optional
message metric `request: {id}`. Each model call has its own assistant message,
including tool follow-ups and retries. Snapshot and event replay preserve this
association. Imported chats discard transient request identities. Neither a
chat ID nor a whole tool run is used as the health join key.

## Presentation and sampling

Use the existing `/api/monitor` SSE history stream and collector. Subscribe while
a model call is active and close when it ends or the view is left. Monitor and
Chat share collection, so opening both does not add inference-server polling.

Match the exact request identity. Use deltas of `prefilled_tokens` and
`published_tokens` over the runtime's `sample_elapsed_ms`. Repeated snapshots,
missing requests, stale data, counter decreases and stream resets do not produce
a rate. A short call with fewer than two usable samples has no live rate. Show
unavailable explicitly and keep final server timings separate from live rates.
Zero is valid only for a measured interval with unchanged counters. Clear live
rates when generation ends instead of holding the last nonzero value.

Keep layout stable through phase changes. Show prompt progress and phase while
prefilling, measured generation rate while generating, and compact final usage
after completion. Preserve readable mobile layout and both themes.

The shared `tools/health_dashboard/request_rates.js` owns the existing interval
and request-rate calculations for both callers. Monitor retains its presentation
policy. Chat clears a current rate immediately on repeated timestamps and closes
its stream when the tab is hidden, the view changes or the model call ends.

## Implementation and checks

The API owns the shared request identity. `tests/server.py` checks the opening
role chunk, exact concurrent health joins, restart separation, cancellation and
unchanged streamed content and usage. The check runs in the ordinary server
component as well as the tensor-group runner.

The companion owns per-call storage and presentation. Engine/store tests cover
bounded first-event IDs, tool rounds, persistence, replay and import removal.
Browser and shared-rate checks cover normal, stale, missing, reset and completed
states, subscription lifetime, mobile layout and both themes. The isolated local
preview also passed correlation against a real runtime reply.

Landing requires CTest, the CPU suite, Qwen3-0.6B byte identity, hosted checks and
the MI50 server component. The hosted HF job runs the full CPU/server suite with
pinned model fixtures. The redundant local full server run was stopped and is
not counted as a pass. Final qualification results are in STATUS.

The runtime identity and companion consumer ship together. Existing upstreams
without the shared identity remain usable, with live rates unavailable. Do not
claim live correlation on a deployment until its upstream supplies the identity.
