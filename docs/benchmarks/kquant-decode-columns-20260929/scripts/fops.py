#!/usr/bin/env python3
# fops.py FILE...: the float multiplies, multiply-adds, fused, adds and shuffled adds of a disassembly, as backend-vulkan counts them.
import re, sys
for f in sys.argv[1:]:
    n = dict(mul=0, mad=0, fused=0, add=0, lane_add=0)
    for line in open(f):
        s = line.strip()
        if not s.startswith('v_'): continue
        op = re.match(r'[A-Za-z0-9_]+', s).group(0)
        low = s.lower()
        if ((op.startswith('v_fma') or op.startswith('v_fmac')) and 'f32' in op) or op.startswith('v_mad_mix_f32'): n['fused'] += 1
        elif op.startswith(('v_mad_f32', 'v_mac_f32', 'v_mad_legacy_f32', 'v_mac_legacy_f32')): n['mad'] += 1
        elif op.startswith(('v_mul_f32', 'v_mul_legacy_f32')): n['mul'] += 0 if '0x4f7ffffe' in low else 1
        elif op.startswith('v_add_f32'): n['lane_add' if '_dpp' in op else 'add'] += 1
    print(f.split('/')[-1], n)
