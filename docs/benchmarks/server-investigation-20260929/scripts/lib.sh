#!/bin/bash
# lib.sh: sourced by a lane script, which sets LANE (a name), O (its output directory), SMI (rocm-smi card ids) and RENDERS (their renderD numbers, comma separated) first.
# Arms, each writing O/NAME-L.json and .out per level, O/NAME.levels (level, CLOCK_MONOTONIC start and end), O/NAME.server.log and a progress line:
#   llmx_arm NAME TREE MODEL "LEVELS" "EXTRA" "ENVS": TREE/b/llmx serve --max-seqs 64 --threads 6 EXTRA over every card of the lane (vulkan:0..N-1), one server for all levels; ENVS are VAR=VALUE words, and a trace arm (LLMX_PROBE_TRACE=trace) writes O/NAME.trace.
#   ref_arm NAME MODEL "LEVELS" "EXTRA" NP: mx-llama.cpp's ROCm llama-server, -ngl 99 -fa 1 -lm dio EXTRA, -sm layer over several cards; NP "users" starts a server per level with -np equal to the users and -c 1152 a slot, a number one server for all levels.
# Every container runs on CPUS (default 4-7,12-15) with --init and no core files, named llmx-p2-perf2-LANE-NAME[-L][-srv].
W=/opt/claude-work/llmx-p2-perf2
CPUS=${CPUS:-4-7,12-15}
TOOL=$W/main/tools/server_load.py
IMG=llmx-p2-dev:vulkan-ccache
REF=mxxm/mx-llama.cpp:gfx906-b10951-eefc4e732
mkdir -p $O
ndev=$(echo $RENDERS | tr ',' '\n' | wc -l)
DLIST=""; for ((k=0;k<ndev;k++)); do DLIST="$DLIST${DLIST:+,}vulkan:$k"; done
HIPV=""; for ((k=0;k<ndev;k++)); do HIPV="$HIPV${HIPV:+,}$k"; done
DARGS=""; for d in ${RENDERS//,/ }; do DARGS="$DARGS --device /dev/dri/renderD$d"; done
GROUPS_="--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)"
VOLS="-v $W:$W -v /opt/claude-work/llmx-vulkan-mi50/models:/m:ro -v $W/models:/m2:ro"

model_path() {
  case $1 in
    q8) echo /m/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf;;
    q4k) echo /m/models--lmstudio-community--Qwen3-8B-GGUF/snapshots/07ebe812301319d9947477e3a94ab8aa587bb3af/Qwen3-8B-Q4_K_M.gguf;;
    q6k) echo /m2/Qwen3-8B-Q6_K.gguf;;
    q40) echo /m2/Qwen3-8B-Q4_0.gguf;;
  esac
}

progress() { echo "$(date +%F\ %T) $LANE $*" >> $O/progress; }

# The load tool's commands for LEVELS against PORT with API, each level timed on CLOCK_MONOTONIC into NAME.levels; llmx's health after each level.
load_cmds() {
  local name=$1 port=$2 api=$3 levels=$4 out=""
  for L in $levels; do
    out="$out t0=\$(python3 -c 'import time;print(time.monotonic())'); python3 $TOOL --url http://127.0.0.1:$port --api $api --concurrency $L --input-len 128 --output-len 128 --rounds 2 --json $O/$name-$L.json > $O/$name-$L.out 2>&1; t1=\$(python3 -c 'import time;print(time.monotonic())'); echo $L \$t0 \$t1 >> $O/$name.levels;"
    [ $api = llmx ] && out="$out curl -s http://127.0.0.1:$port/v1/health > $O/$name-$L.health;"
  done
  echo "$out"
}

llmx_arm() {
  local name=$1 tree=$2 model=$3 levels=$4 extra=$5 envs=$6
  local M=$(model_path $model) T=$W/$tree port=$((20000 + RANDOM % 5000))
  local cname=llmx-p2-perf2-$LANE-$name
  local envx=""
  for e in $envs; do
    case $e in LLMX_PROBE_TRACE=trace) envx="$envx LLMX_PROBE_TRACE=$O/$name.trace";; *) envx="$envx $e";; esac
  done
  progress "$name start $(cut -d' ' -f1-3 /proc/loadavg)"
  rm -f $O/$name.levels $O/$name.trace $O/$name.trace.names
  local ready="for i in \$(seq 900); do curl -sf http://127.0.0.1:$port/v1/health > /dev/null && break; sleep 1; done;"
  timeout 7200 docker run --rm --init --name $cname --cpuset-cpus $CPUS --ulimit core=0 $DARGS $GROUPS_ $VOLS -w /tmp $IMG bash -c \
    "ulimit -c 0; $T/b/llmx --version; sha256sum $T/b/llmx; env $envx $T/b/llmx serve $M --host 127.0.0.1 --port $port --device $DLIST --max-seqs 64 --threads 6 $extra > $O/$name.server.log 2>&1 & s=\$!; $ready $(load_cmds $name $port llmx "$levels") kill \$s; wait \$s" > $O/$name.log 2>&1 || docker stop -t 5 $cname > /dev/null 2>&1
  progress "$name end $(cut -d' ' -f1-3 /proc/loadavg)"
}

ref_one() {
  local name=$1 M=$2 levels=$3 extra=$4 np=$5 tag=$6
  local port=$((25000 + RANDOM % 5000)) ctx=$(( np * 1152 ))
  local cname=llmx-p2-perf2-$LANE-$name$tag
  local sm=""; [ $ndev -gt 1 ] && sm="-sm layer"
  docker run -d --init --name $cname-srv --cpuset-cpus $CPUS --ulimit core=0 --device /dev/kfd $DARGS $GROUPS_ --ipc=host --security-opt seccomp=unconfined \
    -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e GPU_MAX_HW_QUEUES=8 -e HIP_VISIBLE_DEVICES=$HIPV $VOLS -w /tmp --entrypoint bash $REF \
    -c "ulimit -c 0; exec /usr/local/bin/llama-server -m $M -ngl 99 $sm -fa 1 -np $np -c $ctx -lm dio $extra --host 127.0.0.1 --port $port > $O/$name$tag.server.log 2>&1" > /dev/null
  local ready="for i in \$(seq 900); do curl -sf http://127.0.0.1:$port/health > /dev/null && break; sleep 1; done;"
  timeout 7200 docker run --rm --init --name $cname --cpuset-cpus $CPUS --ulimit core=0 --network container:$cname-srv -v $W:$W -w $W $IMG bash -c \
    "ulimit -c 0; $ready $(load_cmds $name $port completion "$levels") true" >> $O/$name.log 2>&1 || docker stop -t 5 $cname > /dev/null 2>&1
  docker stop -t 10 $cname-srv > /dev/null 2>&1
  docker rm -f $cname-srv > /dev/null 2>&1
}

ref_arm() {
  local name=$1 model=$2 levels=$3 extra=$4 np=$5
  local M=$(model_path $model)
  progress "$name start $(cut -d' ' -f1-3 /proc/loadavg)"
  rm -f $O/$name.levels
  if [ "$np" = users ]; then
    for L in $levels; do ref_one $name $M "$L" "$extra" $L -$L; done
  else
    ref_one $name $M "$levels" "$extra" $np ""
  fi
  progress "$name end $(cut -d' ' -f1-3 /proc/loadavg)"
}

clocks_high() {
  for g in $SMI; do rocm-smi -d $g --setperflevel high >> $O/clocks.log 2>&1; done
  trap "for g in $SMI; do rocm-smi -d \$g --setperflevel auto >> $O/clocks.log 2>&1; done; find $W -maxdepth 3 -type f \( -name core -o -name 'core.[0-9]*' \) -delete 2>/dev/null" EXIT
}
