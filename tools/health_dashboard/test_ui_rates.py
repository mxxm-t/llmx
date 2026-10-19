"""Focused, dependency-free checks for dashboard throughput windows."""

import html
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HTML_FILE = Path(__file__).resolve().with_name("index.html")
CHROME_PATHS = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)
CHROME_COMMANDS = ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge")
FUNCTIONS = ("consecutiveCounterRate", "sampleIntervalRate", "sampleIntervalMeasurement", "requestWindowRate", "runtimeSampleAge",
             "resetLiveRequestSample", "updateLiveRequests")
PRELUDE = ("const state = {interval: 2, liveRequests: {available: null, items: []}}; "
           "let fakePerformanceNow = 0; const performance = {now: () => fakePerformanceNow}; "
           "let liveRequestSample = null, liveRequestHistory = [], liveRequestUptime = null; "
           "function number(v) { return typeof v === 'number' && Number.isFinite(v); } "
           "function get(o, path) { return path.split('.').reduce((a,k)=>a&&typeof a==='object'?a[k]:undefined,o); }\n")
CASES = r"""
const outcomes = [];
function check(name, ok) { outcomes.push({name, ok: !!ok}); }
function sample(t, pp, tg, reset = false) {
  return {t, reset, metrics: {
    'requests.live.sample_elapsed_ms': t * 1000,
    'requests.live.work.prefilled_tokens': pp,
    'requests.live.work.generated_tokens': tg
  }};
}
const aggregate = [sample(10, 0, 0), sample(15, 512, 0), sample(20, 512, 512)];
check('burst PP uses the latest adjacent sample interval',
  Math.abs(sampleIntervalRate(aggregate, 2, 'requests.live.work.prefilled_tokens', 'requests.live.sample_elapsed_ms') - 0) < 0.001);
check('burst TG uses the latest adjacent sample interval',
  Math.abs(sampleIntervalRate(aggregate, 2, 'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') - 102.4) < 0.001);
const repeated = [...aggregate, {...sample(22, 512, 512), metrics: aggregate[2].metrics}];
check('duplicate snapshot preserves only a recent measured value',
  Math.abs(sampleIntervalRate(repeated, 3, 'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') - 102.4) < 0.001 &&
  runtimeSampleAge(repeated, 'requests.live.sample_elapsed_ms') === 2);
const oldSnapshot = sample(10, 0, 0);
const freshSnapshot = sample(32, 20, 20);
const gapBetweenOldPolls = [oldSnapshot, {...sample(30, 0, 0), metrics: oldSnapshot.metrics}, freshSnapshot];
check('a real collection gap within the prior snapshot interval invalidates the rate',
  sampleIntervalRate(gapBetweenOldPolls, 2, 'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') === null);
const continuousOldPolls = [oldSnapshot];
for (let t = 12; t <= 30; t += 2) continuousOldPolls.push({...sample(t, 0, 0), metrics: oldSnapshot.metrics});
continuousOldPolls.push(freshSnapshot);
check('continuous duplicate polls still recover on the next runtime snapshot',
  Math.abs(sampleIntervalRate(continuousOldPolls, continuousOldPolls.length - 1,
    'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') - 20 / 22) < 0.001);
check('old duplicate snapshot is unavailable',
  sampleIntervalRate([aggregate[2], {...aggregate[2], t: 40}], 1,
    'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') === null);
check('unchanged distinct samples are a real zero',
  sampleIntervalRate([aggregate[2], sample(30, 512, 512)], 1,
    'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') === 0);
check('collector gaps do not bridge aggregate windows',
  sampleIntervalRate([aggregate[0], {...aggregate[2], t: 40}], 1,
    'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') === null);
check('counter decreases invalidate an aggregate window',
  sampleIntervalRate([sample(10, 0, 0), sample(15, 600, 600), sample(20, 512, 512)], 2,
    'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') === null);
check('server resets invalidate an aggregate window',
  sampleIntervalRate([sample(10, 0, 0), sample(15, 0, 0, true), sample(20, 1, 1)], 2,
    'requests.live.work.generated_tokens', 'requests.live.sample_elapsed_ms') === null);
const request = (timestamp, pp, tg, observedAt, present = true) => ({timestamp, observedAt,
  items: new Map(present ? [['r-1', {prefilled_tokens: pp, published_tokens: tg}]] : [])});
const requestHistory = [request(10000, 0, 0, 10), request(15000, 512, 0, 15), request(20000, 512, 512, 20)];
check('first request sample is unavailable', requestWindowRate(requestHistory.slice(0, 1), 'r-1', 'prefilled_tokens', 2) === null);
check('request PP shares aggregate sample interval',
  Math.abs(requestWindowRate(requestHistory, 'r-1', 'prefilled_tokens', 2) - 0) < 0.001);
check('request TG shares aggregate sample interval',
  Math.abs(requestWindowRate(requestHistory, 'r-1', 'published_tokens', 2) - 102.4) < 0.001);
const boundarySamples = [sample(0, 0, 0), sample(2, 0, 0), sample(4, 100, 100), sample(6, 100, 100),
  sample(8, 100, 100), sample(10, 100, 100), sample(12, 200, 200)];
const boundaryWindow = sampleIntervalMeasurement(boundarySamples, 6,
  'requests.live.work.prefilled_tokens', 'requests.live.sample_elapsed_ms');
const sparseRequests = [request(0, 0, 0, 0), request(4000, 100, 100, 4), request(6000, 100, 100, 6),
  request(8000, 100, 100, 8), request(12000, 200, 200, 12)];
check('request without the exact previous aggregate endpoint is unavailable',
  requestWindowRate(sparseRequests, 'r-1', 'prefilled_tokens', 2, boundaryWindow) === null);
const alignedRequests = [request(0, 0, 0, 0), request(2000, 0, 0, 2), request(4000, 100, 100, 4),
  request(6000, 100, 100, 6), request(8000, 100, 100, 8), request(10000, 100, 100, 10),
  request(12000, 200, 200, 12)];
check('request and aggregate use identical selected endpoints',
  Math.abs(requestWindowRate(alignedRequests, 'r-1', 'prefilled_tokens', 2, boundaryWindow) - boundaryWindow.rate) < 0.001);
const shortAggregateWindow = sampleIntervalMeasurement([sample(0, 0, 0), sample(2, 10, 10)], 1,
  'requests.live.work.prefilled_tokens', 'requests.live.sample_elapsed_ms');
check('aligned request uses the consecutive interval without a warmup window',
  requestWindowRate([request(0, 0, 0, 0), request(2000, 10, 10, 2)], 'r-1', 'prefilled_tokens', 2,
    shortAggregateWindow) === shortAggregateWindow.rate);
resetLiveRequestSample();
updateLiveRequests(liveHealth(10000, 0, 0), null, true);
updateLiveRequests(liveHealth(15000, 512, 0), null, true);
updateLiveRequests(liveHealth(20000, 512, 512), null, true);
check('aligned mode stays unavailable when aggregate endpoints are unavailable',
  state.liveRequests.items[0].pp === null && state.liveRequests.items[0].tg === null);
resetLiveRequestSample();
for (const [seconds, value] of [[0, 0], [2, 0], [4, 100], [6, 100], [8, 100], [10, 100], [12, 200]]) {
  updateLiveRequests(liveHealth(seconds * 1000, value, value),
    {prefilled_tokens: boundaryWindow, generated_tokens: boundaryWindow}, true)
}
check('live request cards use exact aggregate window endpoints',
  Math.abs(state.liveRequests.items[0].pp - boundaryWindow.rate) < 0.001 &&
  Math.abs(state.liveRequests.items[0].tg - boundaryWindow.rate) < 0.001);
resetLiveRequestSample();
fakePerformanceNow = 0;
updateLiveRequests(liveHealth(10000, 0, 0));
for (let seconds = 2; seconds <= 20; seconds += 2) {
  fakePerformanceNow = seconds * 1000;
  updateLiveRequests(liveHealth(10000, 0, 0));
}
check('continuous duplicate polls remain in bounded request history', liveRequestHistory.length === 11);
fakePerformanceNow = 22000;
updateLiveRequests(liveHealth(30000, 100, 100), {
  prefilled_tokens: {startClock: 10000, endClock: 30000},
  generated_tokens: {startClock: 10000, endClock: 30000}
}, true);
check('request rates recover after a long unchanged runtime snapshot',
  state.liveRequests.items[0].pp === 5 && state.liveRequests.items[0].tg === 5);
resetLiveRequestSample();
fakePerformanceNow = 0;
updateLiveRequests(liveHealth(10000, 0, 0));
fakePerformanceNow = 2000;
updateLiveRequests(liveHealth(10000, 0, 0));
fakePerformanceNow = 20000;
updateLiveRequests(liveHealth(10000, 0, 0));
check('a real polling gap clears duplicate snapshot history', liveRequestHistory.length === 0);
fakePerformanceNow = 22000;
updateLiveRequests(liveHealth(30000, 100, 100), {
  prefilled_tokens: {startClock: 10000, endClock: 30000},
  generated_tokens: {startClock: 10000, endClock: 30000}
}, true);
check('rates do not bridge a real polling gap', state.liveRequests.items[0].pp === null && state.liveRequests.items[0].tg === null);
check('new request waits for a second distinct sample',
  requestWindowRate(requestHistory.slice(2), 'r-1', 'prefilled_tokens', 2) === null);
check('vanished request cannot bridge a missing row',
  requestWindowRate([request(10000, 0, 0, 10), request(15000, 0, 0, 15, false), request(20000, 10, 10, 20)],
    'r-1', 'prefilled_tokens', 2) === null);
check('request observation gaps invalidate the window',
  requestWindowRate([request(10000, 0, 0, 1), request(15000, 100, 100, 30)],
    'r-1', 'prefilled_tokens', 2) === null);
check('distinct unchanged request counters remain zero',
  requestWindowRate([request(10000, 0, 0, 1), request(15000, 0, 0, 2), request(20000, 0, 0, 3)],
    'r-1', 'published_tokens', 2) === 0);
function liveHealth(timestamp, pp, tg, id = 'r-1') {
  return {requests: {live: {sample_elapsed_ms: timestamp, limit: 64, total: id ? 1 : 0, omitted: 0,
    items: id ? [{id, phase: 'prefill', prompt_tokens: 500, reused_tokens: 0,
      prefilled_tokens: pp, published_tokens: tg, elapsed_ms: timestamp}] : []}}};
}
resetLiveRequestSample();
updateLiveRequests(liveHealth(10000, 0, 0));
check('live request UI waits for a second distinct sample', state.liveRequests.items[0].pp === null);
updateLiveRequests(liveHealth(15000, 512, 0));
updateLiveRequests(liveHealth(20000, 512, 512));
check('live request UI uses the same sample interval',
  state.liveRequests.items[0].pp === 0 && Math.abs(state.liveRequests.items[0].tg - 102.4) < 0.001);
updateLiveRequests(liveHealth(20000, 512, 512));
check('live request UI keeps last value with stale marker', state.liveRequests.stale && state.liveRequests.items[0].pp === 0);
liveRequestSample.observedAt -= 20;
updateLiveRequests(liveHealth(20000, 512, 512));
check('live request UI expires stale values', state.liveRequests.items[0].pp === null);
resetLiveRequestSample();
updateLiveRequests(liveHealth(1000, 0, 0));
liveRequestSample.observedAt -= 30;
updateLiveRequests(liveHealth(3000, 100, 100));
check('live request UI breaks rates across collector gaps', state.liveRequests.items[0].pp === null);
resetLiveRequestSample();
updateLiveRequests(liveHealth(1000, 0, 0));
updateLiveRequests(liveHealth(2000, 0, 0, null));
updateLiveRequests(liveHealth(3000, 100, 100));
check('live request UI breaks rates when an ID disappears', state.liveRequests.items[0].pp === null);
const legacy = [
  {t: 10, metrics: {'work.since_start.generated_tokens': 0}},
  {t: 15, metrics: {'work.since_start.generated_tokens': 0}},
  {t: 20, metrics: {'work.since_start.generated_tokens': 100}}
];
check('legacy live aggregate uses adjacent collector samples',
  Math.abs(sampleIntervalRate(legacy, 2, 'work.since_start.generated_tokens', null) - 20) < 0.001);
const legacyLong = [
  {t: 100, metrics: {'work.since_start.generated_tokens': 0}},
  {t: 110, metrics: {'work.since_start.generated_tokens': 100}},
  {t: 120, metrics: {'work.since_start.generated_tokens': 110}}
];
check('legacy rate uses consecutive collector samples',
  Math.abs(sampleIntervalRate(legacyLong, 2, 'work.since_start.generated_tokens', null) - 1) < 0.001);
globalThis.rateTestResults = outcomes;
"""

RENDER_CASES = r"""
const outcomes = [];
function check(name, ok) { outcomes.push({name, ok: !!ok}); }
const now = Math.floor(Date.now() / 1000);
const metric = 'requests.live.work.generated_tokens';
const clockField = 'requests.live.sample_elapsed_ms';
const sample = (t, value, reset = false) => ({t: now + t, reset, metrics: {
  [metric]: value, 'requests.live.work.prefilled_tokens': value, [clockField]: (100 + t) * 1000
}});
state.latest = {requests: {live: {sample_elapsed_ms: 110000, work: {prefilled_tokens: 100, generated_tokens: 100}}}};
state.lastSuccess = now;
state.ok = true;
state.received = true;
const item = {id: 'r-1', phase: 'generate', elapsed: 12345, pp: 10, tg: 10, prompt: 100, reused: 0, prefilled: 100, published: 100};
const measures = [];
for (const [name, samples, value, stale] of [
  ['busy', [sample(-10, 0), sample(0, 100)], 10, false],
  ['zero', [sample(-10, 0), sample(0, 0)], 0, false],
  ['stale', [sample(-12, 0), sample(-2, 100), {...sample(0, 100), metrics: sample(-2, 100).metrics}], 10, true],
  ['expired', [sample(-30, 0), sample(-20, 100), {...sample(0, 100), metrics: sample(-20, 100).metrics}], null, true],
  ['reset', [sample(-10, 100), sample(0, 0, true)], null, false],
  ['missing', [sample(-10, 0), sample(0, null)], null, false],
  ['recovery', [sample(-10, 100), sample(0, 200)], 10, false]
]) {
  state.samples = samples;
  state.liveRequests = {available: true, items: [{...item, pp: value, tg: value}], total: 1, omitted: 0, stale, sampleAgeMs: stale ? 2000 : 0};
  render();
  const row = document.querySelector('.live-request-row');
  const strong = row.querySelector('.live-request-rate strong');
  measures.push({name, boxes: [row, $('prompt-chart'), $('throughput-chart'), $('live-requests-card'), $('token-rate-note')].map(node => {
    const r = node.getBoundingClientRect(); return [r.x, r.y, r.width, r.height];
  }), font: getComputedStyle(strong).fontSize, color: getComputedStyle(strong).color});
  if (name === 'zero') check('zero readout is numeric', $('throughput-value').textContent === '0' && strong.textContent === '0');
  if (name === 'expired') check('expired gap reason is visible', $('throughput-at').textContent.includes('unchanged'));
  if (name === 'reset') check('reset gap reason is visible', $('throughput-at').textContent.includes('reset'));
  if (name === 'missing') check('missing counter reason is visible', $('throughput-at').textContent.includes('missing'));
}
check('live card and chart geometry stays fixed through rate states', measures.every(m => JSON.stringify(m.boxes) === JSON.stringify(measures[0].boxes)));
check('live rate type stays fixed through rate states', measures.every(m => m.font === measures[0].font));
check('stale rate retains measured color', measures[2].color === measures[0].color);
check('dashboard fits the viewport without horizontal overflow', document.documentElement.scrollWidth <= innerWidth);
const config = chartConfigurations.find(c => c.id === 'throughput');
renderChart([sample(-10, 0), sample(0, 0), sample(2, null)], now + 2, config, []);
check('isolated known zero has a visible chart marker', !!$('throughput-chart').querySelector('circle[data-sample-value="0"]'));
check('missing point is not drawn as zero', $('throughput-chart').querySelectorAll('circle[data-sample-value]').length === 1);
renderChart([sample(-12, 0), sample(-2, 0), sample(0, null), sample(2, 0), sample(12, 0)], now + 12, config, []);
const d = $('throughput-chart').querySelector('path').getAttribute('d');
check('missing data splits the path between known zeros', (d.match(/M/g) || []).length === 2);
check('recovered historical gap remains visibly shaded', !!$('throughput-chart').querySelector('rect[data-rate-gap]') && $('throughput-chart').textContent.includes('Shaded = unavailable'));
const idleStart = sample(-20, 100);
const idleLatest = {...idleStart, t: now, metrics: {...idleStart.metrics, 'requests.now.active': 0}};
renderChart([sample(-30, 0), idleStart, idleLatest], now, config, []);
check('idle frozen snapshot is explained without inventing zero', $('throughput-at').textContent === 'No active requests / no new runtime snapshot' && $('throughput-value').textContent === '--');
globalThis.rateTestResults = outcomes;
globalThis.layoutMeasurements = measures;
"""


def function_source(source, name):
    match = re.search(r"\bfunction\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", source)
    if not match:
        raise AssertionError("missing JavaScript helper: " + name)
    start = match.start()
    depth = 0
    quote = None
    escaped = False
    for index in range(match.end() - 1, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError("unterminated JavaScript helper: " + name)


class DashboardRateWindowTests(unittest.TestCase):
    def run_browser(self, document, width=1440):
        browser = next((str(path) for path in CHROME_PATHS if path.is_file()), None)
        if browser is None:
            browser = next((shutil.which(command) for command in CHROME_COMMANDS if shutil.which(command)), None)
        if browser is None:
            self.skipTest("Chrome/Edge is required for the focused JavaScript check")
        with tempfile.TemporaryDirectory(prefix="llmx-dashboard-rates-") as temp:
            page = Path(temp) / "rates.html"
            profile = Path(temp) / "profile"
            page.write_text(document, encoding="utf-8", newline="")
            result = subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run",
                                     "--disable-background-networking", "--window-size=" + str(width) + ",1200",
                                     "--user-data-dir=" + str(profile), "--dump-dom", page.as_uri()],
                                    capture_output=True, text=True, timeout=25, check=True)
        match = re.search(r"<pre id=[\"']rate-test-results[\"']>(.*?)</pre>", result.stdout, re.S)
        self.assertIsNotNone(match, "browser did not return JavaScript results")
        outcomes = json.loads(html.unescape(match.group(1)))
        failed = [item["name"] for item in outcomes if not item["ok"]]
        self.assertEqual(failed, [], "failed browser assertions: " + ", ".join(failed))
        return outcomes

    def test_rendered_rate_geometry_and_zero_gaps(self):
        source = HTML_FILE.read_text(encoding="utf-8")
        source = source.rsplit("poll();", 1)[0] + source.rsplit("poll();", 1)[1]
        results = ("<pre id='rate-test-results'></pre><script>" + RENDER_CASES +
                   "\ndocument.querySelector('#rate-test-results').textContent = JSON.stringify(globalThis.rateTestResults);</script>")
        document = source.replace("</body>", results + "</body>")
        for width in (1440, 390):
            with self.subTest(width=width):
                self.assertEqual(len(self.run_browser(document, width)), 13)

    def test_shared_runtime_and_request_windows(self):
        source = HTML_FILE.read_text(encoding="utf-8")
        helpers = "\n".join(function_source(source, name) for name in FUNCTIONS)
        document = ("<!doctype html><html><body><script>" + PRELUDE + helpers + "\n" + CASES +
                    "</script><pre id='rate-test-results'></pre><script>document.querySelector('#rate-test-results').textContent = "
                    "JSON.stringify(globalThis.rateTestResults)</script></body></html>")
        outcomes = self.run_browser(document)
        self.assertEqual(len(outcomes), 34)

    def test_health_contract_and_labels_are_source_backed(self):
        source = HTML_FILE.read_text(encoding="utf-8")
        self.assertIn("requests.live.work.", source)
        self.assertIn("requests.live.sample_elapsed_ms", source)
        self.assertIn("Consecutive collector samples", source)
        self.assertIn("config.fields.map(path => sampleIntervalRate(samples, i, path, config.clockField))", source)
        self.assertIn("requestWindowRate(liveRequestHistory", source)
        self.assertIn("showing last measured sample interval as stale", source)
        self.assertIn("Runtime snapshot unchanged for", source)
        self.assertIn("Repeated snapshots show the last measured rate as stale", source)
        self.assertIn("same consecutive distinct runtime snapshots", source)


if __name__ == "__main__":
    unittest.main()
