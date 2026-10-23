# llmx health dashboard

A standalone, read-only view of an llmx server's `/v1/health` endpoint.
The Python standard library collects one shared sample every two seconds and
keeps up to one hour in memory. Browsers use the collector; opening another tab
does not add another upstream sampler. No inference request is sent.

## Run

From the repository root, with Python 3.12 or later:

```sh
python tools/health_dashboard/server.py --upstream http://127.0.0.1:8090/v1/health --host 127.0.0.1 --port 8182
```

Open `http://127.0.0.1:8182`. Use `--host` to select a trusted LAN address.
This separate tool has no authentication and exposes the same aggregate
operational data as the configured health endpoint. It does not expose prompts
or responses, modify llmx, or start a model.

`--interval` selects 1 to 60 seconds between polls; `--history` selects at most
3,600 seconds of retained data. Data is lost when the dashboard restarts.
The collector requests only the configured URL, refuses redirects, and uses
normal certificate validation for HTTPS. Runtime configuration uses flags.

## Reading the charts

- With `requests.live.work`, prompt and output panels use its
  `prefilled_tokens` and `generated_tokens` counters for aggregate runtime rates.
  Prompt work excludes cache hits, background jobs and speculative work, but
  includes recovery re-prefill. Output counts tokens published to request
  channels, including the first and accepted speculative tokens. Older live
  APIs with `work.since_start` use the explicitly labeled collector-time
  aggregate fallback. The oldest APIs use completed-request counters, which
  advance at request completion and include cached prompt tokens. Missing live
  metrics remain unavailable. Lifetime totals beneath each rate panel follow
  that panel's counter; the selected-window summary continues to label
  completed-request counters separately.
- Both rate panels have independent scales. With `requests.live.work`, aggregate
  PP/TG and per-request PP/TG use cumulative counters from the same consecutive
  distinct runtime snapshots. Each rate is the counter change divided by the
  snapshots' actual elapsed time; per-request rates use those exact same
  endpoints and remain unavailable if either is missing or the request is
  absent in either endpoint. The dashboard does not interpolate or poll more
  often. Repeated snapshots retain the last measured interval as stale briefly
  in the headline and request cards, then become unavailable. Counter resets
  and collector gaps are unavailable rather than zero. A new per-request rate
  becomes available after two distinct snapshots. Old APIs retain the
  collector-time aggregate fallback, which can differ from per-request runtime
  rates. Long passes reveal prompt work only at retirement; zero does not
  establish that the engine was idle. Collection remains shared, every two
  seconds by default, without per-token polling.
- When `requests.live` is present, the live-request cards show the capped
  request IDs, phase, elapsed time and prominent PP/TG rates. Expand a card's
  token-counter details for prompt/reused/prefilled/published counts.
  Rates use counter changes across consecutive distinct runtime samples and
  their `sample_elapsed_ms` timestamps, not browser poll timing. A new or reset
  request has unavailable rates until two distinct snapshots are available;
  unchanged counters across a fresh interval produce a real zero rate. A repeated
  runtime snapshot is shown as stale. The `prefill` phase means no output has
  been generated yet; it does not prove the scheduler is currently executing
  that request. The card list shows how many live requests the API omitted.
  Requests that start and finish between polls may not appear because the
  endpoint exposes current live rows, not completed request history. It contains
  no prompt text or token IDs.
- Rate values keep the same type and spacing as snapshots become stale or
  rates become unavailable. A separate status line labels the last measured
  value; unavailable values use `--`. Known zero rates stay at zero on the
  charts, including isolated observations. Missing counters, expired runtime
  snapshots, resets and collection gaps remain shaded gaps, with the reason
  beside the chart readout and in timeline inspection. Shading remains visible
  in history after collection recovers. An expired runtime snapshot with no
  active requests is labeled as such; an instantaneous active count does not
  establish a measured zero rate for the window.
- Active and reusable device KV tokens are separate reservations reported by
  llmx. They are not measured GPU memory usage; the endpoint has no GPU
  utilization, physical VRAM use or per-request latency distribution.
- Host payload, allocated storage and capacity are different quantities.
  Disk occupancy and transfer counters describe the cache tier.
- Draft acceptance and pacing counters describe work since server start.
  Draft acceptance shows the kept / drafted lifetime counts even when no request
  is currently active. Zero observations show `No data`, missing counters show
  `Unavailable`, and zero kept tokens with a positive drafted count show `0%`.
  A selected chart range does not turn lifetime counters into interval totals.
- Missing metrics remain unavailable. Upstream failure keeps the last good
  snapshot with a stale indication. Server resets and counter decreases break
  derived rates instead of creating negative spikes.

The dashboard groups requests, cache tiers, drafting, scheduling and recovery.
The searchable metrics explorer exposes the complete latest health document,
including fields introduced by newer server versions.

Run the focused rate-window regression with
`python -m unittest tools.health_dashboard.test_ui_rates`. When Chrome or Edge
is installed, it uses a local headless browser to execute the dashboard's rate
helpers with a 90-second bound per browser invocation; a timeout fails the test
and is not retried or skipped. Without Chrome or Edge, the browser check is
skipped. This adds no runtime dependency. The same check renders the full page
at desktop and narrow widths and compares live-card and chart geometry across
busy, zero, stale, expired, reset, missing-counter and recovery states.

The model placement panel shows realized topology when `/v1/health` includes
`topology`: split mode, tensor width, layer stages and backend names. Expand
per-layer details for mixer/feed-forward placement and routed-expert streaming
eligibility; eligibility does not mean that streaming is active for the current
request. These are backend assignments, not inferred physical GPU numbers or
VRAM telemetry. Older runtimes without `topology` show placement as unavailable.
The dashboard does not infer placement from device count.

Hover a chart to inspect the same timestamp across all four charts. Click to
hold it, use the arrow keys to move it, or return to the latest sample. The
selected-window summary shows observed counter changes beside lifetime totals;
missing history, resets and recorded disconnects leave these changes unavailable.
Sample boundaries approximate the selected duration without extrapolation; the
tooltip gives their actual times and duration. Recorded collector events appear
as timeline markers.

Detailed cache, drafting and scheduling counters expand separately. The browser
remembers the range, panel choices and compact spacing when local storage is
available; storage restrictions do not prevent telemetry from working.

## Verify

```sh
python -m unittest discover -s tools/health_dashboard -v
```

These local mock-endpoint tests do not need a model or contact production.
Browser checks cover desktop/mobile rendering, range selection, pause/resume,
metrics search, stale data and recovery. They are recorded at the feature
checkpoint; the native runtime is unchanged.

The Python collector owns upstream polling, health-response validation, the shared
bounded sample/event history, and the small browser-facing history endpoint. It
also serves this static page; the browser draws charts and reads the snapshot.
A lean future option is to let the C++ server serve the same static page and
provide its health data directly. Static hosting alone would not replace the
collector's shared polling, bounded history and event tracking. Preserving that
behavior requires either a native shared collector in llmx, or browser-local
polling and history, which is per-tab and lost on reload. This is a design
option, not a current capability or planned rewrite.
