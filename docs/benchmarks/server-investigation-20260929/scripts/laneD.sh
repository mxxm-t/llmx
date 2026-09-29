#!/bin/bash
# laneD.sh MODELS...: one MI50 (rocm-smi GPU[6], renderD128), the profile arms per model, interleaved on the one card:
# main without profiling, the probe traced (--timing and every dispatch timestamped), the probe untimed, the probe untimed with decode rows at extent 64 (the prompt tile),
# the probe traced with each logits row copied to host memory before sampling (the copy timed apart from the draw), main again.
LANE=${LANE:-d}; SMI=${SMI:-6}; RENDERS=${RENDERS:-128}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
ALL="1 8 16 32 64"
for m in "$@"; do
  llmx_arm $m-main main $m "$ALL" "" ""
  llmx_arm $m-trace probe $m "$ALL" "--timing" "LLMX_PROBE_TRACE=trace"
  llmx_arm $m-probe probe $m "$ALL" "" ""
  llmx_arm $m-tile probe $m "$ALL" "" "LLMX_PROBE_DECODE_EXTENT=64"
  llmx_arm $m-copy probe $m "8 32 64" "--timing" "LLMX_PROBE_TRACE=trace LLMX_PROBE_READ=copy"
  llmx_arm $m-main2 main $m "$ALL" "" ""
done
progress "finished"
