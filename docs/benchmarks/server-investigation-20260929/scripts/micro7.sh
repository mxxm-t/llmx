#!/bin/bash
# micro7.sh TAG SMI RENDER: probe7's matmul probe on one card for the fifth decode-tile form (Q8_0 only): its bits against the tile, per layer 0 and 3 and the head (check),
# then an 8B Q8_0 pass's matmuls per column count through the decode path, the tile and the decode-tile forms (time) twice, then timestamped once.
W=/opt/claude-work/llmx-p2-perf2
tag=$1; SMI=$2; RENDERS=$3; LANE=s
O=$W/jobs/$tag
source $W/lib.sh
clocks_high
P=$W/probe7/b/llmx-perf-matmul-probe
run() {   # run NAME CMD
  local name=$1; shift
  progress "$name start"
  timeout 7200 docker run --rm --init --name llmx-p2-perf2-$LANE-$name --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w /tmp $IMG bash -c "ulimit -c 0; sha256sum $P; $*" > $O/$name.log 2>&1 || docker stop -t 5 llmx-p2-perf2-$LANE-$name > /dev/null 2>&1
  progress "$name end"
}
COLS=1,2,4,8,16,32,64
run check-q8-l0 "$P check $(model_path q8) 0 0 > $O/check-q8-l0.txt 2>&1"
run check-q8-l3 "$P check $(model_path q8) 0 3 > $O/check-q8-l3.txt 2>&1"
for r in 1 2; do run time-q8-$r "$P time $(model_path q8) 0 10 $COLS > $O/time-q8-$r.txt 2>&1"; done
run diag-q8 "LLMX_PROBE_DIAG=1 $P time $(model_path q8) 0 10 $COLS > $O/diag-q8.txt 2>&1"
progress finished
