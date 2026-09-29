#!/bin/bash
# witness.sh LANE "SMI" RENDERS MODELS...: the reference server's configuration witness on a lane's cards: per model, the measured default command (-np 16 at 16 users) with -lv 4 added,
# one load round at 16 users, so its log shows the batch, ubatch, slots, devices and, over several cards, whether pipeline parallelism is enabled; the throughput of these runs is not used.
LANE=$1; SMI=$2; RENDERS=$3; shift 3
O=/opt/claude-work/llmx-p2-perf2/jobs/witness-$LANE
source /opt/claude-work/llmx-p2-perf2/lib.sh
clocks_high
for m in "$@"; do
  ref_arm $m-wit $m "16" "-lv 4" users
  ref_arm $m-witb4096 $m "16" "-lv 4 -b 4096 -ub 1024" users
done
progress finished
