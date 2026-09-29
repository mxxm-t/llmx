#!/usr/bin/env python3
# analyze2.py DIR ARM [--json OUT]: the probe trace DIR/ARM.trace cut by DIR/ARM.levels to each level's two load rounds (from DIR/ARM-L.json), and per level:
#   passes and rows a pass; device time a decode pass by stage and class (matmul without the head, the head, attention/KV, other), summed dispatch time against the union of the device's busy intervals;
#   the scheduler thread's time by activity (the sampling pool's run and the draws on its threads apart, and the copy of a row apart from its draw where the arm copies);
#   each device's idle time, each gap attributed to what it waited for (the producing stage, or the host by what the host was doing);
#   a decode pass's critical path (formed to its logits read) against its summed device time;
#   and each receive wait against the producer's completion and the consumer device's activity.
import bisect, collections, json, os, struct, sys

HOST_KINDS = {1: 'round', 2: 'form', 3: 'stage', 4: 'wait_ticket', 5: 'wait_slot', 6: 'wait_staging', 7: 'write', 8: 'logits_wait',
              9: 'copy', 10: 'sample', 11: 'retire', 12: 'reading', 13: 'submit', 14: 'flush', 15: 'room', 16: 'calib',
              17: 'pool_run', 18: 'draw_copy', 19: 'draw_sample', 20: 'step'}


def load(dirn, arm):
    names = {}
    for line in open(os.path.join(dirn, arm + '.trace.names')):
        i, n = line.split()
        names[int(i)] = n
    host, dev = [], collections.defaultdict(list)
    data = open(os.path.join(dirn, arm + '.trace'), 'rb').read()
    for ty, d, kind, a, b, c, t0, t1 in struct.iter_unpack('<BBHIIIQQ', data):
        if ty == 0:
            host.append((t0, t1, kind, a, b, c, d))
        else:
            dev[d].append((t0, t1, kind, a))
    host.sort()
    for d in dev:
        dev[d].sort()
    return names, host, dev


def kclass(name):
    if name.startswith('matmul') or name.startswith('quantize'):
        return 'matmul'
    if 'attention' in name or name.startswith('kv_write') or name.startswith('norm_rope_kv'):
        return 'attention'
    return 'other'


def clip(a, b, lo, hi):
    return max(0, min(b, hi) - max(a, lo))


def windows(dirn, arm):
    out = []
    for line in open(os.path.join(dirn, arm + '.levels')):
        lvl, t0, t1 = line.split()
        lvl = int(lvl)
        t0, t1 = int(float(t0) * 1e9), int(float(t1) * 1e9)
        p = os.path.join(dirn, '%s-%d.json' % (arm, lvl))
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        durs = [r['summary']['duration'] for r in j['levels'][0]['rounds']]
        best = j['levels'][0]['summary']
        # The rounds end just before the tool writes its table and exits; both rounds are taken, the tool's exit and the gaps estimated at 0.1 s.
        end = t1 - int(0.1e9)
        start = max(t0, end - int((sum(durs) + 0.05) * 1e9))
        out.append((lvl, start, end, best))
    return out


def union(iv):
    tot, cur0, cur1 = 0, None, None
    for a, b in sorted(iv):
        if cur1 is None or a > cur1:
            if cur1 is not None:
                tot += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        tot += cur1 - cur0
    return tot


def main():
    dirn, arm = sys.argv[1], sys.argv[2]
    jout = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    names, host, dev = load(dirn, arm)
    host_t0 = [h[0] for h in host]
    ndev = len(dev)
    stage_of = {}
    for d, ds in dev.items():
        c = collections.Counter(x[3] & 0xff for x in ds[:20000] if x[3] != 0xffffffff)
        stage_of[d] = c.most_common(1)[0][0] if c else d
    last_stage = max(stage_of.values()) if stage_of else 0
    tag_end, tag_start = {}, {}
    by_tag = collections.defaultdict(list)
    for d, ds in dev.items():
        for t0, t1, k, tag in ds:
            if tag == 0xffffffff:
                continue
            by_tag[tag].append((t0, t1, k))
            if tag_end.get(tag, 0) < t1:
                tag_end[tag] = t1
            if tag not in tag_start or tag_start[tag] > t0:
                tag_start[tag] = t0
    # The head: every matmul, quantize and reduce dispatch of a pass's last stage after its last norm, the final norm before the output projection.
    head_disp = set()
    for tag, ds in by_tag.items():
        if (tag & 0xff) != last_stage:
            continue
        ds.sort()
        last_norm = max((i for i, d in enumerate(ds) if names.get(d[2], "?").startswith("rms_norm")), default=None)
        if last_norm is None:
            continue
        for t0, t1, k in ds[last_norm + 1:]:
            if kclass(names.get(k, "?")) == "matmul" or names.get(k, "?").startswith("matmul_reduce"):
                head_disp.add((tag, t0))
    dev_s = {d: [x[0] for x in ds] for d, ds in dev.items()}
    dev_max = {d: max([x[1] - x[0] for x in ds] + [0]) for d, ds in dev.items()}

    def dev_busy(dv, t0, t1):
        ds = dev[dv]
        lo, hi = bisect.bisect_left(dev_s[dv], t0 - dev_max[dv]), bisect.bisect_left(dev_s[dv], t1)
        return union([(max(x[0], t0), min(x[1], t1)) for x in ds[lo:hi] if x[1] > t0 and x[0] < t1])

    results = []
    for lvl, a, b, best in windows(dirn, arm):
        R = {'level': lvl}
        W = b - a
        R['window_s'] = W / 1e9
        R['output_tok_s'] = best['output_tok_s']
        print('=== %s level %d users: window %.2f s, output %.1f tok/s (best round), ttft p99 %.0f ms, itl p50 %.1f p99 %.1f ms' % (
            arm, lvl, W / 1e9, best['output_tok_s'], best['ttft_p99'] * 1e3, best['itl_p50'] * 1e3, best['itl_p99'] * 1e3))
        i0 = bisect.bisect_left(host_t0, a - int(2e9))
        hs = [h for h in host[i0:] if h[0] < b and h[1] > a]
        forms = [h for h in hs if h[2] == 2 and a <= h[0] < b]
        dec = [h[4] & 0xffff for h in forms]
        prm = [h[5] & 0xffff for h in forms]
        want = [h[5] >> 16 for h in forms]
        n = max(1, len(forms))
        hist = collections.Counter(dec)
        nd = sum(1 for x in prm if not x)
        print('  passes %d (%.1f/s): with prompt rows %d, decode rows only %d (mean %.1f rows); rows a pass: decode %.1f, prompt %.1f, wanting logits %.1f; decode rows %s' % (
            len(forms), len(forms) / (W / 1e9), len(forms) - nd, nd, sum(d for d, p in zip(dec, prm) if not p) / max(1, nd),
            sum(dec) / n, sum(prm) / n, sum(want) / n, ', '.join('%d:%d' % kv for kv in sorted(hist.items(), key=lambda kv: -kv[1])[:6])))
        R['passes'] = len(forms)
        R['decode_only_passes'] = nd
        # Device time of each pass by its tag: decode-only passes by their row count, passes with prompt rows together.
        fmeta = {h[3]: (h[4] & 0xffff, h[5] & 0xffff, h[0]) for h in forms}
        per = collections.defaultdict(collections.Counter)
        for dv, ds in dev.items():
            for t0, t1, k, tag in ds[bisect.bisect_left(dev_s[dv], a):bisect.bisect_left(dev_s[dv], b)]:
                if t0 < a or t0 >= b or tag == 0xffffffff:
                    continue
                pid = tag >> 8
                if pid in fmeta:
                    cls = 'head' if (tag, t0) in head_disp else kclass(names.get(k, '?'))
                    per[pid][(tag & 0xff, cls)] += t1 - t0
                    per[pid][(tag & 0xff, 'union_iv')] += 0
        groups = collections.defaultdict(list)
        for pid, (dr, pr, _) in fmeta.items():
            if pid in per:
                groups['decode %d rows' % dr if not pr else 'with prompt rows'].append(pid)
        R['device_pass'] = {}
        for g in sorted(groups, key=lambda g: (g.startswith('with'), -len(groups[g]))):
            lst = groups[g]
            if len(lst) < 3 and not g.startswith('with'):
                continue
            tot = collections.Counter()
            for pid in lst:
                tot.update(per[pid])
            stages_ = sorted(set(k[0] for k in tot))
            parts = []
            rec = {}
            for st in stages_:
                m = sum(v for k, v in tot.items() if k[0] == st) / len(lst) / 1e6
                c = {cl: tot[(st, cl)] / len(lst) / 1e6 for cl in ('matmul', 'head', 'attention', 'other')}
                rec['stage%d' % st] = dict(total=m, **c)
                parts.append('stage %d %.2f ms [matmul %.2f, head %.2f, attention/KV %.2f, other %.2f]' % (
                    st, m, c['matmul'], c['head'], c['attention'], c['other']))
            R['device_pass'][g] = dict(passes=len(lst), **rec)
            print('  device time a pass (dispatch sums), %s (%d passes): %s' % (g, len(lst), '; '.join(parts)))
        # Critical path of decode-only passes: formed to the end of the pass's last logits wait, against the summed device time of its dispatches.
        lw_end = collections.defaultdict(int)
        for h in hs:
            if h[2] == 8:
                lw_end[h[3]] = max(lw_end[h[3]], h[1])
        cp, dsum = [], []
        for pid, (dr, pr, tf) in fmeta.items():
            if pr or pid not in lw_end or pid not in per:
                continue
            cp.append(lw_end[pid] - tf)
            dsum.append(sum(v for k, v in per[pid].items()))
        if cp:
            R['critical_path_ms'] = sum(cp) / len(cp) / 1e6
            R['critical_device_sum_ms'] = sum(dsum) / len(dsum) / 1e6
            print('  decode pass critical path (formed to logits read) %.2f ms mean against %.2f ms of its dispatches summed over the stages (%d passes)' % (
                R['critical_path_ms'], R['critical_device_sum_ms'], len(cp)))
        # The scheduler thread.
        acc = collections.Counter()
        st_iv = sorted((h[0], h[1]) for h in hs if h[2] == 3)
        st_s = [x[0] for x in st_iv]

        def in_stage(t0, t1):
            i = bisect.bisect_right(st_s, t0) - 1
            return i >= 0 and st_iv[i][1] >= t1
        pool = collections.Counter()
        for t0, t1, kind, x, y, z, d in hs:
            if kind in (18, 19):
                pool[kind] += clip(t0, t1, a, b)
                continue
            if kind in (4, 5, 6, 7) and not in_stage(t0, t1):
                acc[100 + kind] += clip(t0, t1, a, b)
                continue
            acc[kind] += clip(t0, t1, a, b)
        rounds = acc[1]
        stage = acc[3]
        waits_in_stage = acc[4] + acc[5] + acc[6]
        record = stage - waits_in_stage - acc[7]
        room = acc[15]
        rooms = [(h[0], h[1]) for h in hs if h[2] == 15]
        stages = [(h[0], h[1]) for h in hs if h[2] == 3]
        st_in_room = 0
        si = 0
        for r0, r1 in rooms:
            for s0, s1 in stages[bisect.bisect_left([s[0] for s in stages], r0):]:
                if s0 > r1:
                    break
                if s1 <= r1:
                    st_in_room += clip(s0, s1, a, b)
        assembly = room - st_in_room
        other = rounds - stage - acc[8] - acc[17] - acc[20] - assembly - acc[12] - acc[14]
        f = lambda v: '%5.1f%%' % (100.0 * v / W)
        thread = dict(recording=record, relay_upload=acc[7], receive_wait=acc[4], slot_wait=acc[5], staging_wait=acc[6], logits_wait=acc[8],
                      sampling_pool=acc[17], step=acc[20], assembly=assembly, timing_reads=acc[12], trace_flush=acc[14],
                      other_in_round=other, idle=W - rounds)
        R['thread_share'] = {k: v / W for k, v in thread.items()}
        print('  thread (share of wall): ' + ', '.join('%s %s' % (k.replace('_', ' '), f(v)) for k, v in thread.items()))
        rn = max(1, sum(1 for h in hs if h[2] == 1 and a <= h[0] < b))
        R['rounds'] = rn
        R['round_ms'] = W / rn / 1e6
        R['per_pass_ms'] = dict(pool=acc[17] / n / 1e6, draw_copy_cpu=pool[18] / n / 1e6, draw_sample_cpu=pool[19] / n / 1e6,
                                step=acc[20] / n / 1e6, logits_wait=acc[8] / n / 1e6, recording=record / n / 1e6)
        print('  rounds %d, mean round %.2f ms; per pass: sampling pool run %.2f ms (its draws %.2f ms of CPU copying rows and %.2f ms sampling), step %.2f ms, logits wait %.2f ms, recording %.2f ms' % (
            rn, W / rn / 1e6, acc[17] / n / 1e6, pool[18] / n / 1e6, pool[19] / n / 1e6, acc[20] / n / 1e6, acc[8] / n / 1e6, record / n / 1e6))
        # Host intervals by activity, for attributing device idle time: the most specific activity wins.
        prio = [8, 17, 20, 4, 5, 6, 7, 12, 14, 16]
        act = [(t0, t1, HOST_KINDS[kind]) for t0, t1, kind, x, y, z, d in hs if kind in prio]
        for t0, t1 in rooms:
            act.append((t0, t1, 'assembly+record0'))
        for t0, t1 in stages:
            act.append((t0, t1, 'recording'))
        rnd = sorted((h[0], h[1]) for h in hs if h[2] == 1)
        act.sort()
        act_s = [x[0] for x in act]
        act_max = max([x[1] - x[0] for x in act] + [0])
        rnd_s = [x[0] for x in rnd]
        rnd_max = max([x[1] - x[0] for x in rnd] + [0])
        rank = {k: i for i, k in enumerate([HOST_KINDS[k] for k in prio] + ['recording', 'assembly+record0'])}

        def attribute(g0, g1, out):
            pts = {g0, g1}
            lo, hi = bisect.bisect_left(act_s, g0 - act_max), bisect.bisect_left(act_s, g1)
            cover = [(max(s, g0), min(e, g1), k) for s, e, k in act[lo:hi] if s < g1 and e > g0]
            lo, hi = bisect.bisect_left(rnd_s, g0 - rnd_max), bisect.bisect_left(rnd_s, g1)
            rcov = [(max(s, g0), min(e, g1)) for s, e in rnd[lo:hi] if s < g1 and e > g0]
            for s, e, k in cover:
                pts.add(s); pts.add(e)
            for s, e in rcov:
                pts.add(s); pts.add(e)
            pts = sorted(pts)
            for p0, p1 in zip(pts, pts[1:]):
                mid = (p0 + p1) // 2
                ks = [k for s, e, k in cover if s <= mid < e]
                if ks:
                    out[min(ks, key=lambda k: rank[k])] += p1 - p0
                elif any(s <= mid < e for s, e in rcov):
                    out['round other'] += p1 - p0
                else:
                    out['thread idle'] += p1 - p0

        R['devices'] = {}
        for d in sorted(dev):
            lo_, hi_ = bisect.bisect_left(dev_s[d], a - dev_max[d]), bisect.bisect_left(dev_s[d], b)
            ds = [x for x in dev[d][lo_:hi_] if x[1] > a and x[0] < b]
            busy = union([(max(x[0], a), min(x[1], b)) for x in ds])
            summed = sum(clip(x[0], x[1], a, b) for x in ds)
            cls = collections.Counter()
            for t0, t1, k, tag in ds:
                cls['head' if (tag, t0) in head_disp else kclass(names.get(k, '?'))] += clip(t0, t1, a, b)
            dep = 0
            why = collections.Counter()
            prev_end = a
            for t0, t1, k, tag in ds:
                if t0 > prev_end + 20000:
                    g0, g1 = prev_end, min(t0, b)
                    s = tag & 0xff
                    if tag != 0xffffffff and s > 0:
                        pe = tag_end.get(((tag >> 8) << 8) | (s - 1))
                        if pe is not None and pe > g0:
                            cut = min(pe, g1)
                            dep += cut - g0
                            g0 = cut
                    if g1 > g0:
                        attribute(g0, g1, why)
                prev_end = max(prev_end, t1)
            if b > prev_end + 20000:
                attribute(prev_end, b, why)
            idle = W - busy
            R['devices'][str(d)] = dict(stage=stage_of[d], busy=busy / W, summed=summed / W, matmul=cls['matmul'] / W, head=cls['head'] / W,
                                        attention=cls['attention'] / W, other=cls['other'] / W, idle=idle / W, waiting_previous_stage=dep / W,
                                        idle_by_host={k: v / W for k, v in why.items()})
            print('  device %d (stage %d): busy %s (dispatches summed %s) [matmul %s, head %s, attention/KV %s, other %s], idle %s: waiting on the previous stage %s; host-side %s' % (
                d, stage_of[d], f(busy), f(summed), f(cls['matmul']), f(cls['head']), f(cls['attention']), f(cls['other']), f(idle), f(dep),
                ', '.join('%s %s' % (k, f(v).strip()) for k, v in why.most_common())))
        if ndev > 1:
            wt = [h for h in hs if h[2] == 4 and a <= h[0] < b]
            if wt:
                tot = sum(h[1] - h[0] for h in wt)
                cons_busy = prod_busy = 0
                late = []
                for t0, t1, kind, tk, last, tag, d in wt:
                    s = tag & 0xff
                    cd = [x for x in dev if stage_of[x] == s]
                    pdv = [x for x in dev if stage_of[x] == s - 1] if s > 0 else []
                    for c in cd:
                        cons_busy += dev_busy(c, t0, t1)
                    for p in pdv:
                        prod_busy += dev_busy(p, t0, t1)
                    pe = tag_end.get(((tag >> 8) << 8) | (s - 1)) if s > 0 else None
                    if pe is not None:
                        late.append((t1 - pe) / 1e3)
                late.sort()
                med = late[len(late) // 2] if late else float('nan')
                R['receive'] = dict(count=len(wt), share=tot / W, mean_ms=tot / len(wt) / 1e6, producer_busy=prod_busy / max(1, tot),
                                    consumer_busy=cons_busy / max(1, tot), wake_after_producer_us=med)
                print('  receive ticket waits: %d, %.1f%% of wall, mean %.2f ms; during them the producing device was busy %.0f%% and the consuming device %.0f%% of the wait; host woke %.0f us (median) after the producer finished' % (
                    len(wt), 100.0 * tot / W, tot / len(wt) / 1e6, 100.0 * prod_busy / max(1, tot), 100.0 * cons_busy / max(1, tot), med))
        results.append(R)
    if jout:
        json.dump({'arm': arm, 'levels': results}, open(jout, 'w'), indent=1)


if __name__ == '__main__':
    main()
