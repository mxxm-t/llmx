#!/bin/bash
# CPU bench under MSVC on the Windows machine, run from the branch tree with main built beside it: main (cb) and the branch (cf) as short single-repeat runs, pairs alternating A B then B A; best and median of each arm are compared, with the machine's CPU use sampled beside every run.
set -u
W=$(pwd)
O=$W/perf; rm -rf $O; mkdir -p $O
declare -A EXE=([cb]=../llmx-main/build-cpu/Release/llmx.exe [cf]=build-cpu/Release/llmx.exe)
H=$HOME/.cache/huggingface/hub
M06=$(cygpath -w $H/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf)
M30=$(cygpath -w $HOME/.cache/llmx/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf)
for t in cb cf; do echo "$t $(sha256sum ${EXE[$t]} | cut -c1-16) $(${EXE[$t]} --version)"; done
cpu() { powershell -NoProfile -Command "[int](Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"; }
one() {  # key i tree threads model args...
    local key=$1 i=$2 t=$3 th=$4 m=$5; shift 5
    local l0=$(cpu | tr -d '\r') f=$O/$key.$i.$t.out
    "${EXE[$t]}" bench --model "$m" --device cpu --threads $th --r 1 "$@" > $f 2>&1 < /dev/null
    local rc=$? pp tg
    pp=$(grep -E "bench: pp" $f | sed -E 's/.*pp[0-9]+ +([0-9.]+).*/\1/')
    tg=$(grep -E "bench: tg" $f | sed -E 's/.*tg[0-9]+ +([0-9.]+).*/\1/')
    echo "$key $i $t rc $rc pp $pp tg $tg cpu% before $l0"
}
date
for spec in "06:16:M06:512:128:12" "06t1:1:M06:128:64:8" "30:16:M30:128:32:6" "30t8:8:M30:128:32:6"; do
    IFS=: read key th mv p n pairs <<< "$spec"
    m=${!mv}
    for i in $(seq 1 $pairs); do
        if [ $((i % 2)) -eq 1 ]; then order="cb cf"; else order="cf cb"; fi
        for t in $order; do one $key $i $t $th "$m" --p $p --n $n; done
    done
done
date
