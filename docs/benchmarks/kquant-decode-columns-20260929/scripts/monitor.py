#!/usr/bin/env python3
# monitor.py OUT [PERIOD]: every PERIOD seconds (10) one JSON line: the wall and CLOCK_MONOTONIC time, the load average, the host's CPU busy and iowait shares and each CPU's busy share,
# the disks' read and write MB/s, the ten busiest processes over the period with their container, and every card's use and VRAM share, until killed.
import json, os, re, subprocess, sys, time

out, period = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0


def cpu_times():
    res = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu'):
            f = line.split()
            v = list(map(int, f[1:9]))
            res[f[0]] = (sum(v), v[3] + v[4], v[4])
    return res


def disks():
    r = w = 0
    for line in open('/proc/diskstats'):
        f = line.split()
        name = f[2]
        if re.fullmatch(r'(sd[a-z]+|nvme\d+n\d+|vd[a-z]+)', name):
            r += int(f[5]); w += int(f[9])
    return r * 512, w * 512


def procs():
    res = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            s = open('/proc/%s/stat' % pid).read()
            comm = s[s.index('(') + 1:s.rindex(')')]
            f = s[s.rindex(')') + 2:].split()
            res[int(pid)] = (comm, int(f[11]) + int(f[12]))
        except Exception:
            pass
    return res


names = {}


def container(pid):
    try:
        cg = open('/proc/%d/cgroup' % pid).read()
    except Exception:
        return ''
    m = re.search(r'docker[-/]([0-9a-f]{64})', cg)
    if not m:
        return ''
    cid = m.group(1)
    if cid not in names:
        try:
            names[cid] = subprocess.run(['docker', 'inspect', '-f', '{{.Name}}', cid], capture_output=True, text=True, timeout=5).stdout.strip().lstrip('/')
        except Exception:
            names[cid] = cid[:12]
    return names[cid]


def gpus():
    try:
        j = json.loads(subprocess.run(['rocm-smi', '--showuse', '--showmemuse', '--json'], capture_output=True, text=True, timeout=20).stdout)
    except Exception:
        return {}
    res = {}
    for k, v in j.items():
        if k.startswith('card'):
            res[k[4:]] = [v.get('GPU use (%)'), v.get('GPU Memory Allocated (VRAM%)')]
    return res


hz = os.sysconf('SC_CLK_TCK')
c0, d0, p0, t0 = cpu_times(), disks(), procs(), time.monotonic()
while True:
    time.sleep(period)
    c1, d1, p1, t1 = cpu_times(), disks(), procs(), time.monotonic()
    dt = t1 - t0
    busy = {}
    for k in c1:
        tot = c1[k][0] - c0[k][0]
        busy[k] = (100.0 * (tot - (c1[k][1] - c0[k][1])) / tot if tot else 0.0, 100.0 * (c1[k][2] - c0[k][2]) / tot if tot else 0.0)
    top = []
    for pid, (comm, t) in p1.items():
        if pid in p0:
            u = 100.0 * (t - p0[pid][1]) / hz / dt
            if u >= 5.0:
                top.append((u, pid, comm))
    top.sort(reverse=True)
    rec = {
        'wall': time.strftime('%Y-%m-%d %H:%M:%S'), 'mono': t1, 'load': open('/proc/loadavg').read().split()[:3],
        'cpu_busy': round(busy['cpu'][0], 1), 'iowait': round(busy['cpu'][1], 1),
        'per_cpu': [round(busy['cpu%d' % i][0]) for i in range(len(c1) - 1)],
        'disk_read_mb_s': round((d1[0] - d0[0]) / dt / 1e6, 1), 'disk_write_mb_s': round((d1[1] - d0[1]) / dt / 1e6, 1),
        'top': [[round(u), pid, comm, container(pid)] for u, pid, comm in top[:10]],
        'gpu': gpus(),
    }
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    c0, d0, p0, t0 = c1, d1, p1, t1
