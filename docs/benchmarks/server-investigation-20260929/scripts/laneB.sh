#!/bin/bash
# laneB.sh MODELS...: a layer split over two MI50s (rocm-smi GPU[0] and GPU[1], renderD136 and renderD137), llmx main against the reference's layer split per model, interleaved:
# llmx at its default passes (P = S), the reference with -np at the users, its batch and ubatch swept, llmx at P = S + 1, llmx's traced probe (--timing), the reference at -np 64 (the old configuration), llmx again, the reference again.
LANE=${LANE:-b}; SMI=${SMI:-"0 1"}; RENDERS=${RENDERS:-136,137}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
ALL="1 8 16 32 64"; HI="16 32 64"
for m in "$@"; do
  llmx_arm $m-llmx main $m "$ALL" "" ""
  ref_arm $m-ref $m "$ALL" "" users
  ref_arm $m-refb4096u1024 $m "$HI" "-b 4096 -ub 1024" users
  ref_arm $m-refb1024u256 $m "$HI" "-b 1024 -ub 256" users
  llmx_arm $m-llmxp3 main $m "$HI" "--passes 3" ""
  llmx_arm $m-trace probe $m "$ALL" "--timing" "LLMX_PROBE_TRACE=trace"
  ref_arm $m-refnp64 $m "$ALL" "" 64
  llmx_arm $m-llmx2 main $m "$ALL" "" ""
  ref_arm $m-ref2 $m "$ALL" "" users
done
progress "finished"
