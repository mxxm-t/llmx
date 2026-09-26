#!/bin/bash
# Inside the timing container, --cpus 6 with no GPU device: CPU bench of main (cb) and the branch (cf) as short single-repeat runs, pairs alternating A B then B A, started without waiting on the load, which is recorded beside every run; best and median of each arm are compared.
set -u
ulimit -c 0
O=/o
M06=/m/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf
M8=/m/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf
M30=/c/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf
load() { cut -d' ' -f1 /proc/loadavg; }
for t in cb cf; do echo "$t $(sha256sum /bin2/$t/llmx | cut -c1-16) $(/bin2/$t/llmx --version)"; done
one() {  # key i tree threads model args...
    local key=$1 i=$2 t=$3 th=$4 m=$5; shift 5
    local l0=$(load) f=$O/$key.$i.$t.out
    /bin2/$t/llmx bench --model $m --device cpu --threads $th --r 1 "$@" > $f 2>&1 < /dev/null
    local rc=$? pp tg
    pp=$(grep -E "bench: pp" $f | sed -E 's/.*pp[0-9]+ +([0-9.]+).*/\1/')
    tg=$(grep -E "bench: tg" $f | sed -E 's/.*tg[0-9]+ +([0-9.]+).*/\1/')
    echo "$key $i $t rc $rc pp $pp tg $tg load $l0 -> $(load)"
}
date
for spec in "06:6:$M06:512:128:12" "06t1:1:$M06:128:64:12" "8b:6:$M8:128:32:8" "8bt1:1:$M8:64:16:6" "30:6:$M30:128:32:8" "30t1:1:$M30:64:16:6"; do
    IFS=: read key th m p n pairs <<< "$spec"
    for i in $(seq 1 $pairs); do
        if [ $((i % 2)) -eq 1 ]; then order="cb cf"; else order="cf cb"; fi
        for t in $order; do one $key $i $t $th $m --p $p --n $n; done
    done
done
date
python3 - $O <<'PY'
import collections, os, re, statistics, sys
rows = collections.defaultdict(lambda: collections.defaultdict(lambda: {"pp": [], "tg": [], "load": []}))
for line in open(os.path.join(sys.argv[1], "log.txt")):
    m = re.match(r"(\S+) \d+ (cb|cf) rc 0 pp ([0-9.]+) tg ([0-9.]+) load ([0-9.]+) -> ([0-9.]+)", line)
    if m:
        r = rows[m.group(1)][m.group(2)]
        r["pp"].append(float(m.group(3)))
        r["tg"].append(float(m.group(4)))
        r["load"].append((float(m.group(5)) + float(m.group(6))) / 2)
for key, arms in rows.items():
    for what in ("pp", "tg"):
        a, b = arms["cb"][what], arms["cf"][what]
        if a and b:
            print("%-5s %s  main best %.2f median %.2f (n %d)  branch best %.2f median %.2f (n %d)  best %+.1f%% median %+.1f%%  load median %.1f/%.1f"
                  % (key, what, max(a), statistics.median(a), len(a), max(b), statistics.median(b), len(b),
                     100 * (max(b) / max(a) - 1), 100 * (statistics.median(b) / statistics.median(a) - 1),
                     statistics.median(arms["cb"]["load"]), statistics.median(arms["cf"]["load"])))
PY
