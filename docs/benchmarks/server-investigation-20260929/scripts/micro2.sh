#!/bin/bash
# micro2.sh TAG SMI RENDER: the matmul probe (probe2's llmx-perf-matmul-probe) on one card: per file the tile's and the decode tile's bits (check), then an 8B pass's matmuls per column count through the decode path, the tile and the decode tile (time),
# twice in alternating model order, then once with every dispatch timestamped (the kernels' device time).
W=/opt/claude-work/llmx-p2-perf2
tag=$1; SMI=$2; RENDERS=$3; LANE=m
O=$W/jobs/$tag
source $W/lib.sh
clocks_high
P=$W/probe2/b/llmx-perf-matmul-probe
run() {   # run NAME CMD
  local name=$1; shift
  progress "$name start"
  timeout 7200 docker run --rm --init --name llmx-p2-perf2-$LANE-$name --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w /tmp $IMG bash -c "ulimit -c 0; sha256sum $P; $*" > $O/$name.log 2>&1 || docker stop -t 5 llmx-p2-perf2-$LANE-$name > /dev/null 2>&1
  progress "$name end"
}
COLS=1,2,4,8,16,32,64
run check-q8 "$P check $(model_path q8) 0 0 > $O/check-q8.txt 2>&1"
run check-q4k-l0 "$P check $(model_path q4k) 0 0 > $O/check-q4k-l0.txt 2>&1"
run check-q4k-l3 "$P check $(model_path q4k) 0 3 > $O/check-q4k-l3.txt 2>&1"
run check-q6k "$P check $(model_path q6k) 0 0 > $O/check-q6k.txt 2>&1"
run check-q40 "$P check $(model_path q40) 0 0 > $O/check-q40.txt 2>&1"
for r in 1 2; do
  order="q8 q4k q6k q40"; [ $r = 2 ] && order="q40 q6k q4k q8"
  for m in $order; do run time-$m-$r "$P time $(model_path $m) 0 10 $COLS > $O/time-$m-$r.txt 2>&1"; done
done
for m in q8 q4k q6k q40; do run diag-$m "LLMX_PROBE_DIAG=1 $P time $(model_path $m) 0 10 $COLS > $O/diag-$m.txt 2>&1"; done
progress finished
