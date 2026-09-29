#!/bin/bash
# slib.sh: sourced by a lane script, which sets LANE, O, SMI and RENDERS (comma separated renderD numbers) first.
# llmx_arm NAME TREE MODEL "LEVELS" "EXTRA" "ENVS": TREE/b/llmx serve --max-seqs 64 --threads 6 EXTRA over the lane's cards, one server for all levels, tools/server_load.py from main,
# closed loop, 128-token prompts and replies with the end of text ignored, greedy, two rounds a level; NAME-L.json/.out per level and NAME.levels (level, CLOCK_MONOTONIC start and end).
W=/opt/claude-work/llmx-p2-kqcols
CPUS=${CPUS:-4-7,12-15}
TOOL=$W/main/tools/server_load.py
IMG=llmx-p2-dev:vulkan-ccache
mkdir -p $O
ndev=$(echo $RENDERS | tr ',' '\n' | wc -l)
DLIST=""; for ((k=0;k<ndev;k++)); do DLIST="$DLIST${DLIST:+,}vulkan:$k"; done
DARGS=""; for d in ${RENDERS//,/ }; do DARGS="$DARGS --device /dev/dri/renderD$d"; done
GROUPS_="--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)"
VOLS="-v $W:$W -v /opt/claude-work/llmx-vulkan-mi50/models:/m:ro -v /opt/claude-work/llmx-p2-perf2/models:/m2:ro"
model_path() {
  case $1 in
    q8) echo /m/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf;;
    q4k) echo /m/models--lmstudio-community--Qwen3-8B-GGUF/snapshots/07ebe812301319d9947477e3a94ab8aa587bb3af/Qwen3-8B-Q4_K_M.gguf;;
    q6k) echo /m2/Qwen3-8B-Q6_K.gguf;;
    q40) echo /m2/Qwen3-8B-Q4_0.gguf;;
  esac
}
progress() { echo "$(date +%F\ %T) $LANE $*" >> $O/progress; }
load_cmds() {
  local name=$1 port=$2 levels=$3 out=""
  for L in $levels; do
    out="$out t0=\$(python3 -c 'import time;print(time.monotonic())'); python3 $TOOL --url http://127.0.0.1:$port --api llmx --concurrency $L --input-len 128 --output-len 128 --rounds 2 --json $O/$name-$L.json > $O/$name-$L.out 2>&1; t1=\$(python3 -c 'import time;print(time.monotonic())'); echo $L \$t0 \$t1 >> $O/$name.levels;"
    out="$out curl -s http://127.0.0.1:$port/v1/health > $O/$name-$L.health;"
  done
  echo "$out"
}
llmx_arm() {
  local name=$1 tree=$2 model=$3 levels=$4 extra=$5 envs=$6
  local M=$(model_path $model) T=$W/$tree port=$((20000 + RANDOM % 5000))
  local cname=llmx-p2-kqcols-$LANE-$name
  local envx=""; for e in ${envs//,/ }; do envx="$envx $e"; done
  progress "$name start $(cut -d' ' -f1-3 /proc/loadavg)"
  rm -f $O/$name.levels
  local ready="for i in \$(seq 900); do curl -sf http://127.0.0.1:$port/v1/health > /dev/null && break; sleep 1; done;"
  timeout 7200 docker run --rm --init --name $cname --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w /tmp $IMG bash -c \
    "ulimit -c 0; $T/b/llmx --version; sha256sum $T/b/llmx; env $envx $T/b/llmx serve $M --host 127.0.0.1 --port $port --device $DLIST --max-seqs 64 --threads 6 $extra > $O/$name.server.log 2>&1 & s=\$!; $ready $(load_cmds $name $port "$levels") kill \$s; wait \$s" > $O/$name.log 2>&1 || docker stop -t 5 $cname > /dev/null 2>&1
  progress "$name end $(cut -d' ' -f1-3 /proc/loadavg)"
}
clocks_high() {
  for g in $SMI; do rocm-smi -d $g --setperflevel high >> $O/clocks.log 2>&1; done
  trap "for g in $SMI; do rocm-smi -d \$g --setperflevel auto >> $O/clocks.log 2>&1; done; find $W -maxdepth 4 -type f \( -name core -o -name 'core.[0-9]*' \) -delete 2>/dev/null" EXIT
}
