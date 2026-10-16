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

- When health exposes `work.since_start`, prompt and output panels use its
  `prefilled_tokens` and `generated_tokens` counters for live aggregate
  wall-clock throughput over the actual interval between shared samples.
  Prompt work excludes cache hits, background jobs and speculative work, but
  includes recovery re-prefill. Output counts tokens published to request
  channels, including the first and accepted speculative tokens.
- Older APIs use an explicitly labeled completed-work fallback. Those request
  counters advance at request completion, and their prompt totals include cached
  tokens. Missing live metrics remain unavailable. Lifetime totals beneath each
  rate panel follow that panel's counter; the selected-window summary continues
  to label completed-request counters separately.
- Both rate panels have independent scales. Live rates are aggregate throughput,
  not per-request or per-kernel speeds. Long passes reveal prompt work only at
  retirement; zero does not establish that the engine was idle. Collection
  remains shared, every two seconds by default, without per-token polling.
- Active and reusable device KV tokens are separate reservations reported by
  llmx. They are not measured GPU memory usage; the endpoint has no GPU
  utilization, physical VRAM use or per-request latency distribution.
- Host payload, allocated storage and capacity are different quantities.
  Disk occupancy and transfer counters describe the cache tier.
- Draft acceptance and pacing counters describe work since server start.
  A selected chart range does not turn lifetime counters into interval totals.
- Missing metrics remain unavailable. Upstream failure keeps the last good
  snapshot with a stale indication. Server resets and counter decreases break
  derived rates instead of creating negative spikes.

The dashboard groups requests, cache tiers, drafting, scheduling and recovery.
The searchable metrics explorer exposes the complete latest health document,
including fields introduced by newer server versions.

The model placement panel shows the runtime-reported split mode, tensor width,
layer stages and the backend names assigned to each stage. Expand per-layer
details for mixer/feed-forward placement and routed-expert streaming eligibility.
Those are realized backend assignments, not inferred GPU numbers or physical
VRAM telemetry. Older health APIs without `topology` show placement as
unavailable; the dashboard does not infer it from the device count.

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
