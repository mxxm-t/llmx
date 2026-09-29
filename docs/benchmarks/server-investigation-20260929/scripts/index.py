#!/usr/bin/env python3
# index.py DIR: DIR/index.json listing every file under DIR with its size, SHA-256 and what it holds.
import hashlib, json, os, sys

D = sys.argv[1]
WHAT = [
    ('README.md', 'the record: method, tables, findings and the ranked fixes'),
    ('summary.txt', 'every server and level of every lane: best round, both rounds, TTFT and ITL p50/p99, failures, configuration witness'),
    ('summary.json', 'the same as JSON'),
    ('flags.txt', 'every timed level against the monitor samples inside it, with the activity flags of scripts/FLAGS.txt'),
    ('flags.json', 'the same as JSON'),
    ('previous-20260928.txt', 'the servers of 2026-09-28 (main 3da159b9 and the reference at -np 64 and -np = users), from their load records'),
    ('previous-20260928.json', 'the same as JSON'),
    ('binaries.sha256', 'SHA-256 of every llmx and probe binary that ran'),
    ('models.sha256', 'SHA-256 of the four model files'),
    ('raw.tar.xz', "every lane directory but the load-tool JSON (tables, server logs, level windows, health, progress, clocks, the mix checks' ids), the witness and microbench runs, the build and fetch logs and the monitor samples; no dispatch traces"),
    ('records.tar.xz', "the load tool's JSON records of every level and request, inter-token gaps included, every time rounded to 0.1 ms"),
    ('analysis/', 'per traced arm: analyze2.py output, text and JSON'),
    ('micro/microtab.md', 'the probe tool\'s time lines per model, path and column count: mean, min, max and runs'),
    ('micro/microtab.json', 'the same with every run and its file'),
    ('micro/micro/', 'first microbench run (probe2): checks of the tile and the first decode-tile form, timings, timestamped timings'),
    ('micro/micro3/', 'probe3: checks and timings with the second form'),
    ('micro/micro4/', 'probe4: checks and timings with the third form'),
    ('micro/micro5/', 'probe5: the decode class with parts of 16 and 32 blocks, checks and one round of timings (stopped after it)'),
    ('micro/micro6/', 'probe6: checks and timings with the fourth form'),
    ('micro/micro7/', 'probe7: checks and timings with the fifth form (Q8_0)'),
    ('scripts/', 'the lane, microbench, witness, build, fetch and packaging scripts, the monitor, the flag definitions, and the analysis and table scripts'),
    ('probe-src/', 'each probe tree as a diff against 6147753a, cumulative from probe.diff (trace) to probe7.diff (every decode-tile form), and probe8.diff (form 2 with the attention guard keyed on the decode extent)'),
]
files = []
for root, _, names in os.walk(D):
    for n in sorted(names):
        p = os.path.join(root, n)
        rel = os.path.relpath(p, D).replace(os.sep, '/')
        if rel == 'index.json':
            continue
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        what = next((w for k, w in WHAT if rel == k or (k.endswith('/') and rel.startswith(k))), '')
        files.append({'path': rel, 'bytes': os.path.getsize(p), 'sha256': h, 'what': what})
files.sort(key=lambda f: f['path'])
json.dump({'record': 'server speed investigation, 2026-09-29, main 6147753a', 'files': files}, open(os.path.join(D, 'index.json'), 'w'), indent=1)
print(len(files), 'files')
