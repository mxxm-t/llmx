#!/usr/bin/env python3
# tworow_check.py DIR: the two-row pair rule of backend-vulkan's --isa screen, run on a dumped representation directory.
import glob, os, re, sys, subprocess
d = sys.argv[1]
def ops(f):
    out = subprocess.run(['python3', os.path.join(os.path.dirname(__file__), 'fops.py'), f], capture_output=True, text=True).stdout
    return eval(out.split(' ', 1)[1])
kinds = ['mul', 'mad', 'fused', 'add', 'lane_add']
builds = {}
for f in glob.glob(os.path.join(d, 'matmul_row_*col.txt')):
    first = open(f).readline()
    m = re.match(r'; row_build cols=(\d+) rows=(\d+)', first)
    if not m: continue
    name = os.path.basename(f)[:-4]
    kernel = re.sub(r'_\d+col$', '', name)
    builds.setdefault(kernel, []).append((int(m.group(1)), int(m.group(2)), ops(f)))
ok = True
for k, bs in sorted(builds.items()):
    ref = ops(os.path.join(d, k + '_1col.txt'))
    bs.sort()
    pers = set()
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            ca, ra, ga = bs[i]; cc, rc, gc = bs[j]
            step = ra * (cc - ca)
            per = []
            for kk in kinds:
                dlt = gc[kk] - ga[kk]
                if dlt < 0 or dlt % step or dlt // step > ref[kk]: print('FAIL', k, ca, cc, kk, ga[kk], gc[kk], ref[kk]); ok = False
                per.append(dlt // step if step else 0)
            if per[1] * ref['fused'] != per[2] * ref['mad']: print('FAIL fused proportion', k); ok = False
            pers.add(tuple(per))
    print(k, 'builds', [b[0] for b in bs], 'per column', pers, 'one-column', ref)
    if len(pers) > 1: print('FAIL inconsistent', k); ok = False
print('ok' if ok else 'FAILED')
