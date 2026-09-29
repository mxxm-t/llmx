#!/bin/bash
# micro.sh TAG SMI RENDER MODEL ROUNDS ARM...: on one card, each arm's probe check (bits against the column alone and a digest of the columns alone), then its decode timing,
# ROUNDS rounds in alternating arm order. An arm is NAME=TREE[:VAR=VALUE,...]. Writes jobs/TAG/NAME-check.txt, NAME-time-R.txt and a CLOCK_MONOTONIC window per run in runs.txt.
W=/opt/claude-work/llmx-p2-kqcols
tag=$1; SMI=$2; RENDER=$3; model=$4; rounds=$5; shift 5
O=$W/jobs/$tag; mkdir -p $O
CPUS=${CPUS:-4-7,12-15}
IMG=llmx-p2-dev:vulkan-ccache
GROUPS_="--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)"
VOLS="-v $W:$W -v /opt/claude-work/llmx-vulkan-mi50/models:/m:ro -v /opt/claude-work/llmx-p2-perf2/models:/m2:ro"
case $model in
  q8) M=/m/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf;;
  q4k) M=/m/models--lmstudio-community--Qwen3-8B-GGUF/snapshots/07ebe812301319d9947477e3a94ab8aa587bb3af/Qwen3-8B-Q4_K_M.gguf;;
  q6k) M=/m2/Qwen3-8B-Q6_K.gguf;;
  q40) M=/m2/Qwen3-8B-Q4_0.gguf;;
esac
COLS=${COLS:-1,2,4,8,16,32,64}
ITERS=${ITERS:-10}
rocm-smi -d $SMI --setperflevel high >> $O/clocks.log 2>&1
trap "rocm-smi -d $SMI --setperflevel auto >> $O/clocks.log 2>&1; find $W -maxdepth 4 -type f \( -name core -o -name 'core.[0-9]*' \) -delete 2>/dev/null" EXIT
mono() { python3 -c 'import time;print(time.monotonic())'; }
run() {   # run NAME TREE ENVS OUTFILE MODE ARGS
  local name=$1 tree=$2 envs=$3 out=$4; shift 4
  local P=$W/$tree/b/llmx-kq-probe envx=""
  for e in ${envs//,/ }; do envx="$envx $e"; done
  local t0=$(mono)
  timeout 3600 docker run --rm --init --name llmx-p2-kqcols-m-$tag-$name --cpuset-cpus $CPUS --ulimit core=0 --device /dev/dri/renderD$RENDER $GROUPS_ $VOLS -w /tmp $IMG bash -c "ulimit -c 0; sha256sum $P; env $envx $P $*" > $O/$out 2>&1 || docker stop -t 5 llmx-p2-kqcols-m-$tag-$name > /dev/null 2>&1
  echo "$out $t0 $(mono)" >> $O/runs.txt
}
arms=("$@")
for a in "${arms[@]}"; do
  name=${a%%=*}; rest=${a#*=}; tree=${rest%%:*}; envs=""; [ "$rest" != "$tree" ] && envs=${rest#*:}
  run $name $tree "$envs" $name-check.txt check $M 0 0
done
for ((r=1; r<=rounds; r++)); do
  order=("${arms[@]}")
  if [ $((r % 2)) = 0 ]; then order=(); for ((i=${#arms[@]}-1; i>=0; i--)); do order+=("${arms[$i]}"); done; fi
  for a in "${order[@]}"; do
    name=${a%%=*}; rest=${a#*=}; tree=${rest%%:*}; envs=""; [ "$rest" != "$tree" ] && envs=${rest#*:}
    run $name $tree "$envs" $name-time-$r.txt time $M 0 $ITERS $COLS
  done
done
echo finished >> $O/runs.txt
