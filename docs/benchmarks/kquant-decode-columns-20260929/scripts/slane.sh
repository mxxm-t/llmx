#!/bin/bash
# slane.sh LANE "SMI" RENDERS MODEL ARM...: servers of each arm (NAME=TREE[:ENVS]) in order on the lane's cards, levels 1 8 16 32 64.
LANE=$1; SMI=$2; RENDERS=$3; model=$4; shift 4
O=/opt/claude-work/llmx-p2-kqcols/jobs/srv-$LANE
source /opt/claude-work/llmx-p2-kqcols/slib.sh
clocks_high
for a in "$@"; do
  name=${a%%=*}; rest=${a#*=}; tree=${rest%%:*}; envs=""; [ "$rest" != "$tree" ] && envs=${rest#*:}
  llmx_arm $name $tree $model "1 8 16 32 64" "" "$envs"
done
progress finished
