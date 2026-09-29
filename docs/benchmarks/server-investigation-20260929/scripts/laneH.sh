#!/bin/bash
# laneH.sh: the stage balance of the two-MI50 layer split (rocm-smi GPU[0] and GPU[1]), llmx main at 16, 32 and 64 users with the layers moved toward stage 0,
# which the traces show idle while stage 1 runs the head and, in Q4_K_M, the heavier Q6_K layers; interleaved with the equal split: equal, moved, moved further, equal.
LANE=${LANE:-h}; SMI=${SMI:-"0 1"}; RENDERS=${RENDERS:-136,137}
O=/opt/claude-work/llmx-p2-perf2/jobs/lane-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
HI="16 32 64"
llmx_arm q8-eq main q8 "$HI" "" ""
llmx_arm q8-s1917 main q8 "$HI" "--layer-shares 19,17" ""
llmx_arm q8-s2016 main q8 "$HI" "--layer-shares 20,16" ""
llmx_arm q8-eq2 main q8 "$HI" "" ""
llmx_arm q4k-eq main q4k "$HI" "" ""
llmx_arm q4k-s2016 main q4k "$HI" "--layer-shares 20,16" ""
llmx_arm q4k-s2115 main q4k "$HI" "--layer-shares 21,15" ""
llmx_arm q4k-eq2 main q4k "$HI" "" ""
# A third round of the split against the reference, with fewer of this investigation's lanes beside it than lane b had.
for m in q8 q6k; do
  llmx_arm $m-llmx3 main $m "$HI" "" ""
  ref_arm $m-ref3 $m "$HI" "" users
  ref_arm $m-refb4096u1024-2 $m "$HI" "-b 4096 -ub 1024" users
done
progress "finished"
