#!/usr/bin/env python3
# report.py JOBS: the comparison tables in Markdown from the lanes' load records (summ2.py's JSON, JOBS/summary.json) and the previous day's (JOBS/previous-20260928.json); prints them for the README.
# Each cell lists every server's best round in run order; the ratio is llmx's best server against the reference's best server of any setting.
import json, os, sys

J = sys.argv[1]
summ = json.load(open(os.path.join(J, 'summary.json')))
prev = json.load(open(os.path.join(J, 'previous-20260928.json')))
MODELS = [('q8', 'Qwen3-8B Q8_0'), ('q4k', 'Qwen3-8B Q4_K_M'), ('q6k', 'Qwen3-8B Q6_K'), ('q40', 'Qwen3-8B Q4_0')]
LEVELS = [1, 8, 16, 32, 64]


def runs(group, L):
    out = []
    for lane, arms in group:
        for a in arms:
            c = summ.get(lane, {}).get(a, {}).get(str(L))
            if c:
                out.append((c['best'], a, c))
    return out


def fmt(rs):
    return ', '.join('%.0f' % r[0] for r in rs) if rs else '-'


def lat(c):
    return '%.0f / %.1f / %.0f' % (c['ttft_p99'], c['itl_p50'], c['itl_p99'])


def prevrow(groups):
    out = {}
    for L in LEVELS:
        vals = []
        for group in groups:
            best = None
            for d, a in group:
                c = prev.get(d, {}).get(a, {}).get(str(L))
                if c and (best is None or c['best'] > best):
                    best = c['best']
            vals.append('%.0f' % best if best else '-')
        out[L] = vals
    return out


def compare(title, llmx, refs, np64, prevrows):
    print('\n#### %s\n' % title)
    print('| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |')
    print('|---:|---|---|---|---:|---|---|---|')
    for L in LEVELS:
        ll = runs(llmx, L)
        rg = [runs(g, L) for g in refs]
        allr = [r for g in rg for r in g]
        n64 = runs(np64, L)
        allr = allr + n64
        if not ll and not allr:
            continue
        bl = max(ll, key=lambda r: r[0]) if ll else None
        br = max(allr, key=lambda r: r[0]) if allr else None
        ratio = '%+.0f%%' % (100.0 * (bl[0] / br[0] - 1)) if bl and br else '-'
        p = prevrows.get(L, ['-', '-', '-'])
        print('| %d | %s | %s | %s | %s | %s | %s | %s |' % (
            L, fmt(ll), ' ; '.join(fmt(g) for g in rg), fmt(n64), ratio, lat(bl[2]) if bl else '-', lat(br[2]) if br else '-', ' ; '.join(p)))


P1 = {'q8': prevrow([[('c1', 'm8'), ('c1', 'm8b'), ('c1n8', 'm8n'), ('c1c', 'm8c')], [('c1', 'r8'), ('c1', 'r8b')], [('c1n8', 'r8n16'), ('c1n8', 'r8n32'), ('c1c', 'r8c32'), ('c1c', 'r8c64')]]),
      'q4k': prevrow([[('c1', 'm4'), ('c1', 'm4b'), ('c1n4', 'm4n')], [('c1', 'r4'), ('c1', 'r4b')], [('c1n4', 'r4n16'), ('c1n4', 'r4n32')]])}
P2 = {'q8': prevrow([[('c2', 'm2a'), ('c2', 'm2b'), ('c2t', 't2')], [('c2', 'r2d')], [('c2t', 'r2n16'), ('c2t', 'r2n32')]]),
      'q4k': prevrow([[('c2q', 'm24'), ('c2q', 'm24b'), ('c2t', 't24')], [('c2q', 'r24d')], [('c2q', 'r24n16'), ('c2q', 'r24n32')]])}
for m, name in MODELS:
    a, f = ('lane-a', 'lane-f') if m in ('q8', 'q4k') else ('lane-c', 'lane-g')
    compare('%s, one MI50' % name,
            [(a, [m + '-llmx', m + '-llmx2']), (f, [m + '-llmx3', m + '-llmx4'])],
            [[(a, [m + '-ref', m + '-ref2']), (f, [m + '-ref3'])], [(a, [m + '-refb4096u1024']), (f, [m + '-refb4096u1024-2'])], [(a, [m + '-refb1024u256'])]],
            [(a, [m + '-refnp64'])], P1.get(m, {}))
    compare('%s, a layer split over two MI50s' % name,
            [('lane-b', [m + '-llmx', m + '-llmx2']), ('lane-h', [m + '-eq', m + '-eq2', m + '-llmx3'])],
            [[('lane-b', [m + '-ref', m + '-ref2']), ('lane-h', [m + '-ref3'])], [('lane-b', [m + '-refb4096u1024']), ('lane-h', [m + '-refb4096u1024-2'])], [('lane-b', [m + '-refb1024u256'])]],
            [('lane-b', [m + '-refnp64'])], P2.get(m, {}))


# Where the time goes: the traced arms' analyses (analyze2.py --json), JOBS/analysis/LANE-ARM.json.
def breakdown(title, path):
    if not os.path.exists(path):
        print('\n#### %s\n\n(no trace)' % title)
        return
    A = json.load(open(path))
    print('\n#### %s\n' % title)
    print('| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |')
    print('|---:|---:|---:|---|---:|---|---|---:|---|')
    for R in A['levels']:
        dp = R['device_pass']
        dec = [(g, v) for g, v in dp.items() if not g.startswith('with')]
        if not dec:
            continue
        g, v = max(dec, key=lambda gv: gv[1]['passes'])
        rows = g.split()[1]
        stages = sorted(k for k in v if k.startswith('stage'))
        parts = '; '.join('%.1f + %.1f + %.1f + %.1f = %.1f' % (v[s]['matmul'], v[s]['head'], v[s]['attention'], v[s]['other'], v[s]['total']) for s in stages)
        tot_dec = sum(gv[1]['passes'] * sum(gv[1][s]['total'] for s in gv[1] if s.startswith('stage')) for gv in dec)
        pr = dp.get('with prompt rows')
        tot_pr = pr['passes'] * sum(pr[s]['total'] for s in pr if s.startswith('stage')) if pr else 0
        share = 100.0 * tot_pr / (tot_pr + tot_dec) if tot_pr + tot_dec else 0
        devs = R['devices']
        busy = ' / '.join('%.0f%%' % (100 * devs[d]['busy']) for d in sorted(devs))
        idle = []
        for d in sorted(devs):
            x = devs[d]
            causes = dict(x['idle_by_host'])
            if x['waiting_previous_stage'] > 0.001:
                causes['previous stage'] = x['waiting_previous_stage']
            top = sorted(causes.items(), key=lambda kv: -kv[1])[:3]
            idle.append('%.0f%%: %s' % (100 * x['idle'], ', '.join('%s %.1f' % (k.replace('_', ' '), 100 * val) for k, val in top)))
        cp = '%.1f / %.1f' % (R.get('critical_path_ms', 0), R.get('critical_device_sum_ms', 0))
        print('| %d | %.0f | %s | %s | %.0f%% | %s | %s | %.1f | %s |' % (R['level'], R['output_tok_s'], rows, parts, share, busy, ' ; '.join(idle), R['round_ms'], cp))


print('\n### Where the time goes')
for m, name in MODELS:
    breakdown('%s, one MI50' % name, os.path.join(J, 'analysis', 'lane-d-%s-trace.json' % m))
    breakdown('%s, a layer split over two MI50s' % name, os.path.join(J, 'analysis', 'lane-b-%s-trace.json' % m))


# The microbench, from micro/microtab.json when given (JOBS/microtab.json).
mt_path = os.path.join(J, 'microtab.json')
if os.path.exists(mt_path):
    mt = json.load(open(mt_path))
    agg = {}
    for r in mt:
        if r['split'] != 'tile split':
            continue
        agg[(r['model'], r['path'], r['cols'])] = sum(r['ms']) / len(r['ms'])
    print('\n### Microbench: decode path, tile, best decode-tile form (ms a pass)\n')
    print('| columns | ' + ' | '.join('%s decode / tile / best form' % n for _, n in MODELS) + ' |')
    print('|---:|' + '---|' * len(MODELS))
    for c in [1, 2, 4, 8, 16, 32, 64]:
        cells = []
        for m, _ in MODELS:
            d = agg.get((m, 'decode', c))
            t = agg.get((m, 'tile', c))
            forms = [(agg[(m, p, c)], p) for p in ('dtile', 'dtile2', 'dtile3', 'dtile4', 'dtile5') if (m, p, c) in agg]
            b = min(forms) if forms else None
            label = {'dtile': 'form 1', 'dtile2': 'form 2', 'dtile3': 'form 3', 'dtile4': 'form 4', 'dtile5': 'form 5'}
            cells.append('%s / %s / %s' % ('%.1f' % d if d else '-', '%.1f' % t if t else '-', '%.1f (%s)' % (b[0], label[b[1]]) if b else '-'))
        print('| %d | %s |' % (c, ' | '.join(cells)))
    print('\n### Decode-tile forms, ms a pass at 1, 8 and 16 columns\n')
    print('| model | columns | decode path | tile | form 1 | form 2 | form 3 | form 4 | form 5 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for m, name in MODELS:
        for c in [1, 8, 16]:
            vals = [agg.get((m, p, c)) for p in ('decode', 'tile', 'dtile', 'dtile2', 'dtile3', 'dtile4', 'dtile5')]
            print('| %s | %d | %s |' % (name, c, ' | '.join('%.1f' % v if v else '-' for v in vals)))


# Served decode tile (lane e) and stage balance (lane h).
def served(lane, arms, title):
    print('\n#### %s\n' % title)
    print('| model | users | ' + ' | '.join(a for a in arms) + ' |')
    print('|---|---:|' + '---|' * len(arms))
    for m, name in MODELS:
        for L in LEVELS:
            cells = []
            for a in arms:
                c = summ.get(lane, {}).get('%s-%s' % (m, a), {}).get(str(L))
                cells.append('%.0f (ITL p50 %.1f)' % (c['best'], c['itl_p50']) if c else '-')
            if any(x != '-' for x in cells):
                print('| %s | %d | %s |' % (name, L, ' | '.join(cells)))


served('lane-e', ['base', 'tile', 'dtile', 'base2'], 'Decode rows through the tile, one MI50 (tok/s)')
served('lane-e2', ['base', 'dtile'], 'The decode tile again on probe8, its attention guard keyed on the decode extent, one MI50 (tok/s)')
served('lane-h', ['eq', 's1917', 's2016', 's2115', 'eq2'], 'Stage balance on two MI50s (tok/s)')


# Flags per lane (JOBS/flags.json).
fl_path = os.path.join(J, 'flags.json')
if os.path.exists(fl_path):
    fl = json.load(open(fl_path))
    print('\n### Flags per lane\n')
    print('| lane | timed levels | flagged cpu | flagged disk | unflagged | host CPU busy, range of samples | this investigation\'s CPU, mean of samples | outside processes seen busiest |')
    print('|---|---:|---:|---:|---:|---|---:|---|')
    for lane in sorted(fl):
        n = c = dk = u = 0
        lo, hi, own = 101, -1, []
        outside = {}
        for arm, lv in fl[lane].items():
            for L, r in lv.items():
                n += 1
                c += 'cpu' in r['flags']
                dk += 'disk' in r['flags']
                u += not r['flags']
                if r['cpu_busy_min'] is not None:
                    lo, hi = min(lo, r['cpu_busy_min']), max(hi, r['cpu_busy_max'])
                if r['own_cpu_pct_mean'] is not None:
                    own.append(r['own_cpu_pct_mean'])
                for k, v in r['outside'].items():
                    outside[k] = max(outside.get(k, 0), v)
        top = sorted(outside.items(), key=lambda kv: -kv[1])[:3]
        print('| %s | %d | %d | %d | %d | %.0f-%.0f%% | %.0f%% | %s |' % (lane, n, c, dk, u, lo, hi, sum(own) / len(own) if own else 0, ', '.join('%s %d%%' % kv for kv in top)))
