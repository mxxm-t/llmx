#!/usr/bin/env python3
# summ2.py DIR... [--json OUT]: every arm and level of each lane directory: the best round's output tok/s, both rounds', TTFT and ITL p50/p99, failures,
# and for the reference its configuration witness from its log (pipeline parallelism, n_batch, n_ubatch, n_seq_max, devices, offload), for llmx its version line and passes.
import glob, json, os, re, sys


def witness(log):
    if not os.path.exists(log):
        return {}
    t = open(log, errors='replace').read()
    w = {}
    m = re.search(r'pipeline parallelism enabled[^\n]*', t)
    if m:
        w['pp'] = m.group(0).strip()
    for key in ('n_batch', 'n_ubatch', 'n_seq_max', 'n_ctx', 'flash_attn', 'n_parallel'):
        m = re.search(r'\b%s\s*=\s*(\S+)' % key, t)
        if m:
            w[key] = m.group(1)
    devs = re.findall(r'(ROCm\d+)[^\n]*?(MI50|MI60|gfx906)[^\n]*', t)
    if devs:
        w['devices'] = sorted(set(d[0] for d in devs))
    m = re.findall(r'using device (\S+)[^\n]*', t)
    if m:
        w['using'] = sorted(set(m))
    m = re.search(r'offloaded (\d+/\d+) layers', t)
    if m:
        w['offloaded'] = m.group(1)
    m = re.search(r'(\d+) pass(?:es)? in flight[^\n]*', t)
    if m:
        w['passes'] = m.group(0)
    return w


out = {}
args = [a for a in sys.argv[1:] if a != '--json']
jpath = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
if jpath:
    args.remove(jpath)
for d in args:
    lane = os.path.basename(d.rstrip('/'))
    print('==', lane)
    arms = {}
    for f in sorted(glob.glob(os.path.join(d, '*-*.json'))):
        base = os.path.basename(f)[:-5]
        arm, lvl = base.rsplit('-', 1)
        if not lvl.isdigit():
            continue
        arms.setdefault(arm, {})[int(lvl)] = f
    for arm in sorted(arms):
        rows = {}
        for lvl in sorted(arms[arm]):
            j = json.load(open(arms[arm][lvl]))
            L = j['levels'][0]
            s = L['summary']
            rounds = [r['summary']['output_tok_s'] for r in L['rounds']]
            fails = sum(r['summary']['failed'] for r in L['rounds'])
            wlog = os.path.join(d, '%s-%d.server.log' % (arm, lvl))
            wit = witness(wlog) if os.path.exists(wlog) else witness(os.path.join(d, arm + '.server.log'))
            rows[lvl] = dict(best=s['output_tok_s'], rounds=rounds, ttft_p50=s['ttft_p50'] * 1e3, ttft_p99=s['ttft_p99'] * 1e3,
                             itl_p50=s['itl_p50'] * 1e3, itl_p99=s['itl_p99'] * 1e3, failed=fails, short=s['short'], witness=wit)
            print('  %-22s %3d users  best %7.1f tok/s  rounds %-15s ttft p50/p99 %6.0f/%6.0f ms  itl p50/p99 %6.1f/%6.1f ms  fail %d short %d  %s' % (
                arm, lvl, s['output_tok_s'], '/'.join('%.1f' % x for x in rounds), s['ttft_p50'] * 1e3, s['ttft_p99'] * 1e3,
                s['itl_p50'] * 1e3, s['itl_p99'] * 1e3, fails, s['short'], ' '.join('%s=%s' % kv for kv in wit.items() if kv[0] in ('pp', 'n_batch', 'n_ubatch', 'n_seq_max', 'devices', 'passes'))))
        out.setdefault(lane, {})[arm] = rows
if jpath:
    json.dump(out, open(jpath, 'w'), indent=1)
