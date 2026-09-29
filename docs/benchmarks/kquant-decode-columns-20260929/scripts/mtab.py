#!/usr/bin/env python3
# mtab.py DIR...: each lane's decode pass-matmul ms per arm and column count, every round, and the mean.
import glob, os, re, sys, collections
for d in sys.argv[1:]:
    t = collections.defaultdict(lambda: collections.defaultdict(list))
    arms = []
    for f in sorted(glob.glob(os.path.join(d, '*-time-*.txt'))):
        arm = re.sub(r'-time-\d+\.txt$', '', os.path.basename(f))
        if arm not in arms: arms.append(arm)
        for line in open(f):
            m = re.match(r'time n=(\d+) decode pass-matmuls ([\d.]+) ms', line)
            if m: t[arm][int(m.group(1))].append(float(m.group(2)))
    order = sorted(arms, key=lambda a: (a not in ('base', 'mainp'), a))
    ns = sorted({n for a in t for n in t[a]})
    print('## %s: ms a pass (rounds), mean' % os.path.basename(d.rstrip('/')))
    print('| cols | ' + ' | '.join(order) + ' |')
    print('|---:|' + '---:|' * len(order))
    for n in ns:
        row = []
        for a in order:
            v = t[a][n]
            row.append('%s -> %.1f' % ('/'.join('%.1f' % x for x in v), sum(v) / len(v)) if v else '-')
        print('| %d | %s |' % (n, ' | '.join(row)))
    print()
