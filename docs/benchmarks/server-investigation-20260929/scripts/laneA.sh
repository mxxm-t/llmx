#!/bin/bash
# laneA.sh MODELS...: one MI50 (rocm-smi GPU[4], renderD134), llmx main against the reference per model, interleaved: llmx, reference with -np at the users, its batch and ubatch swept at 16-64 users, the reference at -np 64 (the old configuration), llmx again, the reference again.
LANE=${LANE:-a}; SMI=${SMI:-4}; RENDERS=${RENDERS:-134}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
ALL="1 8 16 32 64"; HI="16 32 64"
for m in "$@"; do
  llmx_arm $m-llmx main $m "$ALL" "" ""
  ref_arm $m-ref $m "$ALL" "" users
  ref_arm $m-refb4096u1024 $m "$HI" "-b 4096 -ub 1024" users
  ref_arm $m-refb1024u256 $m "$HI" "-b 1024 -ub 256" users
  ref_arm $m-refnp64 $m "$ALL" "" 64
  llmx_arm $m-llmx2 main $m "$ALL" "" ""
  ref_arm $m-ref2 $m "$ALL" "" users
done
progress "finished"
