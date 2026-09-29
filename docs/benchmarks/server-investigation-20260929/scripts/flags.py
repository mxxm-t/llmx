#!/usr/bin/env python3
# flags.py MONITOR DIR... [--json OUT]: every timed level of every arm in the lane directories against the monitor's samples inside it, with the activity flags FLAGS.txt fixed before measuring:
#   cpu  - host CPU busy above 60 percent, a process outside this investigation's containers above 100 percent of a CPU, or a CPU of 4-7,12-15 above 90 percent while a process outside it is among the busiest;
#   disk - reads or writes above 100 MB/s.
# It prints each level's samples, the host CPU busy range, this investigation's own CPU share, the busiest outside processes, and its flags.
import collections, glob, json, os, sys

mon = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
monos = [m['mono'] for m in mon]
args = sys.argv[2:]
jout = None
if '--json' in args:
    jout = args[args.index('--json') + 1]
    args = args[:args.index('--json')]
MINE = 'llmx-p2-perf2'
res = {}
for d in args:
    lane = os.path.basename(d.rstrip('/'))
    for lv in sorted(glob.glob(os.path.join(d, '*.levels'))):
        arm = os.path.basename(lv)[:-7]
        for line in open(lv):
            L, t0, t1 = line.split()
            t0, t1 = float(t0), float(t1)
            ss = [m for m in mon if t0 <= m['mono'] <= t1 + 10]
            flags = set()
            outside = collections.Counter()
            mine = []
            for m in ss:
                if m['cpu_busy'] > 60:
                    flags.add('cpu')
                foreign = [t for t in m['top'] if not t[3].startswith(MINE)]
                mine.append(sum(t[0] for t in m['top'] if t[3].startswith(MINE)))
                for t in foreign:
                    outside['%s(%s)' % (t[2], t[3] or 'host')] = max(outside['%s(%s)' % (t[2], t[3] or 'host')], t[0])
                    if t[0] > 100:
                        flags.add('cpu')
                if foreign and any(m['per_cpu'][c] > 90 for c in (4, 5, 6, 7, 12, 13, 14, 15)):
                    flags.add('cpu')
                if m['disk_read_mb_s'] > 100 or m['disk_write_mb_s'] > 100:
                    flags.add('disk')
            busy = [m['cpu_busy'] for m in ss]
            r = dict(samples=len(ss), flags=sorted(flags), cpu_busy_min=min(busy) if busy else None, cpu_busy_max=max(busy) if busy else None,
                     own_cpu_pct_mean=sum(mine) / len(mine) if mine else None, outside=dict(outside.most_common(6)))
            res.setdefault(lane, {}).setdefault(arm, {})[L] = r
            print('%-8s %-22s %3s users: %2d samples, host CPU busy %s-%s%%, own %.0f%% of a CPU, flags %s; outside %s' % (
                lane, arm, L, len(ss), r['cpu_busy_min'], r['cpu_busy_max'], r['own_cpu_pct_mean'] or 0, ','.join(r['flags']) or '-',
                ', '.join('%s %d%%' % kv for kv in outside.most_common(4))))
if jout:
    json.dump(res, open(jout, 'w'), indent=1)
