#!/usr/bin/env python3
# stab.py DIR...: each server arm's output tok/s and ITL p50 per level, both rounds, from the load tool's JSON records.
import glob, json, os, re, sys, collections
for d in sys.argv[1:]:
    res = collections.defaultdict(dict)
    arms = []
    for f in sorted(glob.glob(os.path.join(d, '*-*.json'))):
        m = re.match(r'(.+)-(\d+)\.json$', os.path.basename(f))
        if not m: continue
        arm, L = m.group(1), int(m.group(2))
        if arm not in arms: arms.append(arm)
        try: j = json.load(open(f))
        except Exception as e: res[arm][L] = 'err'; continue
        rounds = []
        for lv in j.get('levels', [j]):
            for r in lv.get('rounds', []):
                s = r.get('summary', r)
                rounds.append((s.get('output_tok_s'), 1000.0 * s['itl_p50'] if s.get('itl_p50') is not None else None))
        res[arm][L] = rounds
    print('## %s' % os.path.basename(d.rstrip('/')))
    Ls = sorted({L for a in res for L in res[a]})
    print('| users | ' + ' | '.join(arms) + ' |')
    print('|---:|' + '---|' * len(arms))
    for L in Ls:
        cells = []
        for a in arms:
            v = res[a].get(L)
            if not v or v == 'err': cells.append(str(v)); continue
            cells.append(', '.join('%s' % (('%.0f' % t) if isinstance(t, (int, float)) else t) for t, _ in v) + (' (ITL p50 %s)' % ('/'.join(('%.1f' % i) if isinstance(i, (int, float)) else str(i) for _, i in v))))
        print('| %d | %s |' % (L, ' | '.join(cells)))
    print()
