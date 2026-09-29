#!/usr/bin/env python3
# microtab.py DIR...: every `time` line of the matmul probe's time-*.txt files in the micro directories, as (model, split, path, columns) -> the ms of each run,
# printed per model as a table of the mean over runs (min-max), and written as JSON with every run beside it.
import collections, glob, json, os, re, sys

runs = collections.defaultdict(list)
for d in sys.argv[1:]:
    if d.endswith('.json'):
        continue
    for f in sorted(glob.glob(os.path.join(d, 'time-*.txt'))):
        base = os.path.basename(f)[5:-4]
        parts = base.split('-')
        model = parts[0]
        split = next((p for p in parts if p.startswith('k')), 'tile split')
        for line in open(f):
            m = re.match(r'time n=(\d+) (\S+) pass-matmuls ([\d.]+) ms', line)
            if m:
                runs[(model, split, m.group(2), int(m.group(1)))].append((float(m.group(3)), os.path.basename(d) + '/' + os.path.basename(f)))
out = [dict(model=k[0], split=k[1], path=k[2], cols=k[3], ms=[v[0] for v in vs], files=[v[1] for v in vs]) for k, vs in sorted(runs.items())]
jp = [a for a in sys.argv[1:] if a.endswith('.json')]
if jp:
    json.dump(out, open(jp[0], 'w'), indent=1)
models = sorted(set(k[0] for k in runs))
for model in models:
    for split in sorted(set(k[1] for k in runs if k[0] == model)):
        paths = [p for p in ['decode', 'tile', 'tile64', 'dtile', 'dtile2', 'dtile3', 'dtile4', 'dtile5'] if any(k[0] == model and k[1] == split and k[2] == p for k in runs)]
        cols = sorted(set(k[3] for k in runs if k[0] == model and k[1] == split))
        print('\n%s, %s: ms a pass (36 layers and the head), mean of runs [min-max]' % (model, split))
        print('| columns | ' + ' | '.join(paths) + ' |')
        print('|---:|' + '---:|' * len(paths))
        for c in cols:
            cells = []
            for p in paths:
                v = [x[0] for x in runs.get((model, split, p, c), [])]
                cells.append('%.2f [%.2f-%.2f] (%d)' % (sum(v) / len(v), min(v), max(v), len(v)) if v else '-')
            print('| %d | %s |' % (c, ' | '.join(cells)))
