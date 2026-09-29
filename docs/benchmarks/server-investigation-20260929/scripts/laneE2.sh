#!/bin/bash
# laneE2.sh: on probe8, whose attention guard keys on the decode extent rather than on a view of one row (lane e's guard sent a prompt's one-row slice to the per-row kernel,
# so a prompt's last row depended on how its prompt was sliced and the mix check failed): the batch-invariance witness of the decode tile again, a control without the decode tile,
# and the decode tile served again at 1, 16 and 64 users. One MI50 (rocm-smi GPU[3], renderD133, on loan).
LANE=${LANE:-e2}; SMI=${SMI:-3}; RENDERS=${RENDERS:-133}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
mix() {   # mix NAME MODEL ENVS
  local name=$1 m=$2 envs=$3
  progress "$name start"
  timeout 3600 docker run --rm --init --name llmx-p2-perf2-$LANE-$name --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w $W/main $IMG bash -c \
    "ulimit -c 0; env $envs python3 tools/server_mix_check.py --exe $W/probe8/b/llmx --model $(model_path $m) --text tests/data/wiki.test.raw --device vulkan:0 --requests 32 --max-seqs 32 --logprobs --cli 0 --ids $O/$name.ids.json" > $O/$name.log 2>&1 || docker stop -t 5 llmx-p2-perf2-$LANE-$name > /dev/null 2>&1
  progress "$name end $(tail -1 $O/$name.log)"
}
mix mix-q8-dtile q8 "LLMX_PROBE_DECODE_EXTENT=64 LLMX_PROBE_DTILE=16"
mix mix-q8-control q8 "LLMX_PROBE_NONE=1"
mix mix-q4k-dtile q4k "LLMX_PROBE_DECODE_EXTENT=64 LLMX_PROBE_DTILE=16"
llmx_arm q8-base probe8 q8 "1 16 64" "" ""
llmx_arm q8-dtile probe8 q8 "1 16 64" "" "LLMX_PROBE_DECODE_EXTENT=64 LLMX_PROBE_DTILE=16"
llmx_arm q4k-dtile probe8 q4k "1 16 64" "" "LLMX_PROBE_DECODE_EXTENT=64 LLMX_PROBE_DTILE=16"
progress finished
