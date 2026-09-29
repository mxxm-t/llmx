#!/bin/bash
# gate_run.sh KIND RENDER ...: one gate job of the branch in the build image on one card.
W=/opt/claude-work/llmx-p2-kqcols
kind=$1; R=$2; shift 2
GR="--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)"
VOLS="-v $W:$W -v /opt/claude-work/llmx-vulkan-mi50/models:/m:ro -v /opt/claude-work/llmx-p2-perf2/models:/m2:ro"
M8=/m/models--lmstudio-community--Qwen3-8B-GGUF/snapshots/07ebe812301319d9947477e3a94ab8aa587bb3af
M06=/m/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b
mkdir -p $W/jobs/gate
case $kind in
ctest) cmd="cd $W/gate && ctest --test-dir b --output-on-failure > $W/jobs/gate/ctest.txt 2>&1; echo exit=\$? >> $W/jobs/gate/ctest.txt; python3 tests/run_tests.py --exe b/llmx --device vulkan:0 > $W/jobs/gate/suite-vk.txt 2>&1; echo exit=\$? >> $W/jobs/gate/suite-vk.txt";;
ident) cmd="$W/gate_ident.sh vulkan:0 $W/jobs/gate/ident-vk; $W/gate_ident.sh cpu $W/jobs/gate/ident-cpu";;
mix) cmd=""; for m in "$@"; do n=${m%%=*}; f=${m#*=}; for arm in mainref gate; do cmd="$cmd python3 $W/gate/tools/server_mix_check.py --exe $W/$arm/b/llmx --model $f --text $W/gate/tests/data/wiki.test.raw --device vulkan:0 --requests 16 --max-seqs 16 --logprobs --ids $W/jobs/gate/mix-$n-$arm.json > $W/jobs/gate/mix-$n-$arm.txt 2>&1; echo exit=\$? >> $W/jobs/gate/mix-$n-$arm.txt;"; done; done;;
esac
timeout 10800 docker run --rm --init --name llmx-p2-kqcols-g-$kind-$R --cpuset-cpus ${CPUS:-4-7,12-15} --ulimit core=0 --device /dev/dri/renderD$R $GR $VOLS -w /tmp llmx-p2-dev:vulkan-ccache bash -c "ulimit -c 0; $cmd" > $W/jobs/gate/$kind-$R.log 2>&1 || docker stop -t 5 llmx-p2-kqcols-g-$kind-$R
find $W -maxdepth 4 -type f \( -name core -o -name 'core.[0-9]*' \) -delete 2>/dev/null
echo done >> $W/jobs/gate/$kind-$R.log
