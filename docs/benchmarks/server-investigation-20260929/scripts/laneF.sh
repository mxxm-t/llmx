#!/bin/bash
# laneF.sh MODELS...: a third interleaved round on one MI50 at 16, 32 and 64 users, since the reference's rounds vary widely: llmx main, the reference with -np at the users, the reference with -b 4096 -ub 1024, llmx main again.
# LANE, SMI and RENDERS name the card (lane f: rocm-smi GPU[4], renderD134; lane g: GPU[5], renderD135).
LANE=${LANE:-f}; SMI=${SMI:-4}; RENDERS=${RENDERS:-134}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
HI="16 32 64"
for m in "$@"; do
  llmx_arm $m-llmx3 main $m "$HI" "" ""
  ref_arm $m-ref3 $m "$HI" "" users
  ref_arm $m-refb4096u1024-2 $m "$HI" "-b 4096 -ub 1024" users
  llmx_arm $m-llmx4 main $m "$HI" "" ""
done
progress "finished"
