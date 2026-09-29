#!/bin/bash
# laneE.sh MODELS...: one MI50 (rocm-smi GPU[3], renderD133, on loan), the decode-tile server arms on probe3's binary, interleaved per model:
# the probe untimed (decode rows as main), decode rows at extent 64 through the prompt tile (attention of one-row views kept on the per-row kernel),
# the same with calls of up to 16 columns through the decode tile (shaders/matmul_dtile2.comp, the tile's bits), and the probe untimed again.
# Then the batch-invariance witness of the decode tile: tools/server_mix_check.py with log-probabilities, 32 requests at up to 32 at once, so passes cross the 16-column switch.
LANE=${LANE:-e}; SMI=${SMI:-3}; RENDERS=${RENDERS:-133}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
ALL="1 8 16 32 64"
for m in "$@"; do
  llmx_arm $m-base probe3 $m "$ALL" "" ""
  llmx_arm $m-tile probe3 $m "$ALL" "" "LLMX_PROBE_DECODE_EXTENT=64"
  llmx_arm $m-dtile probe3 $m "$ALL" "" "LLMX_PROBE_DECODE_EXTENT=64 LLMX_PROBE_DTILE=16"
  llmx_arm $m-base2 probe3 $m "$ALL" "" ""
done
for m in q8 q4k; do
  progress "mix-$m start"
  timeout 3600 docker run --rm --init --name llmx-p2-perf2-$LANE-mix-$m --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w $W/main $IMG bash -c \
    "ulimit -c 0; LLMX_PROBE_DECODE_EXTENT=64 LLMX_PROBE_DTILE=16 python3 tools/server_mix_check.py --exe $W/probe3/b/llmx --model $(model_path $m) --text tests/data/wiki.test.raw --device vulkan:0 --requests 32 --max-seqs 32 --logprobs --cli 0 --ids $O/mix-$m.ids.json" > $O/mix-$m.log 2>&1 || docker stop -t 5 llmx-p2-perf2-$LANE-mix-$m > /dev/null 2>&1
  progress "mix-$m end $(tail -1 $O/mix-$m.log)"
done
progress finished
