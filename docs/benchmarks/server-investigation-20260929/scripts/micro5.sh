#!/bin/bash
# micro5.sh TAG SMI RENDER: probe5's matmul probe on one card with the decode class's own inner-dimension split: for parts of 16 and of 32 quant blocks ($LLMX_PROBE_DTILE_KPER),
# the decode tile against the tile at that split, bit for bit, per file (check), then an 8B pass's matmuls per column count (time), twice in alternating split order, then timestamped once at 16.
W=/opt/claude-work/llmx-p2-perf2
tag=$1; SMI=$2; RENDERS=$3; LANE=q
O=$W/jobs/$tag
source $W/lib.sh
clocks_high
P=$W/probe5/b/llmx-perf-matmul-probe
run() {   # run NAME ENV CMD
  local name=$1 envs=$2; shift 2
  progress "$name start"
  timeout 7200 docker run --rm --init --name llmx-p2-perf2-$LANE-$name --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w /tmp $IMG bash -c "ulimit -c 0; sha256sum $P; env $envs $*" > $O/$name.log 2>&1 || docker stop -t 5 llmx-p2-perf2-$LANE-$name > /dev/null 2>&1
  progress "$name end"
}
COLS=1,2,4,8,16,32,64
for k in 16 32; do
  run check-q8-k$k LLMX_PROBE_DTILE_KPER=$k "$P check $(model_path q8) 0 0 > $O/check-q8-k$k.txt 2>&1"
  run check-q4k-k$k LLMX_PROBE_DTILE_KPER=$k "$P check $(model_path q4k) 0 0 > $O/check-q4k-k$k.txt 2>&1"
  run check-q6k-k$k LLMX_PROBE_DTILE_KPER=$k "$P check $(model_path q6k) 0 0 > $O/check-q6k-k$k.txt 2>&1"
  run check-q40-k$k LLMX_PROBE_DTILE_KPER=$k "$P check $(model_path q40) 0 0 > $O/check-q40-k$k.txt 2>&1"
done
for r in 1 2; do
  ks="16 32"; [ $r = 2 ] && ks="32 16"
  for k in $ks; do
    for m in q8 q4k q6k q40; do run time-$m-k$k-$r LLMX_PROBE_DTILE_KPER=$k "$P time $(model_path $m) 0 10 $COLS > $O/time-$m-k$k-$r.txt 2>&1"; done
  done
done
for m in q8 q4k q6k q40; do run diag-$m-k16 "LLMX_PROBE_DTILE_KPER=16 LLMX_PROBE_DIAG=1" "$P time $(model_path $m) 0 10 $COLS > $O/diag-$m-k16.txt 2>&1"; done
progress finished
