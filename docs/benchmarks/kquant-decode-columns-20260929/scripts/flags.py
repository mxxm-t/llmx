#!/usr/bin/env python3
# flags.py MONITOR CORES DIR...: every timed server level of the lane directories against the monitor's samples inside it, with the flags of FLAGS.txt; CORES the lane's cores, e.g. 4-7,12-15.
import glob, json, os, sys
mon = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cores = []
for part in sys.argv[2].split(','):
    a, _, b = part.partition('-'); cores += list(range(int(a), int(b or a) + 1))
MINE = 'llmx-p2-kqcols'
for d in sys.argv[3:]:
    for lv in sorted(glob.glob(os.path.join(d, '*.levels'))):
        arm = os.path.basename(lv)[:-7]
        for line in open(lv):
            L, t0, t1 = line.split(); t0, t1 = float(t0), float(t1)
            ss = [m for m in mon if t0 <= m['mono'] <= t1 + 10]
            fl, busy, outside = set(), [], {}
            for m in ss:
                busy.append(m['cpu_busy'])
                if m['cpu_busy'] > 60: fl.add('cpu')
                foreign = [t for t in m['top'] if not t[3].startswith(MINE)]
                for t in foreign:
                    k = '%s(%s)' % (t[2], t[3] or 'host'); outside[k] = max(outside.get(k, 0), t[0])
                    if t[0] > 100: fl.add('cpu')
                if foreign and any(m['per_cpu'][c] > 90 for c in cores): fl.add('cpu')
                if m['disk_read_mb_s'] > 100 or m['disk_write_mb_s'] > 100: fl.add('disk')
            top = sorted(outside.items(), key=lambda kv: -kv[1])[:3]
            print('%s %s L=%s samples=%d busy=%s flags=%s outside=%s' % (os.path.basename(d), arm, L, len(ss), ('%.0f-%.0f' % (min(busy), max(busy))) if busy else '-', ','.join(sorted(fl)) or 'none', top))
