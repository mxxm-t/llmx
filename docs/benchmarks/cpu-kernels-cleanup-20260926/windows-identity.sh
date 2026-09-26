#!/bin/bash
# CLI identity on the CPU under MSVC, run from the branch tree with main built beside it: main (cb) against the branch (cf), name rc(main,branch) hash(main) hash(branch).
set -u
W=$(pwd)
O=$W/id; rm -rf $O; mkdir -p $O
CB=../llmx-main/build-cpu/Release/llmx.exe
CF=build-cpu/Release/llmx.exe
H=$HOME/.cache/huggingface/hub
M06=$H/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf
U=$H/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b
M06Q4=$U/Qwen3-0.6B-Q4_0.gguf
M06Q5=$U/Qwen3-0.6B-Q5_K_M.gguf
M30=$HOME/.cache/llmx/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf
head -c 1500 tests/data/wiki.test.raw > $O/excerpt.txt
head -c 4000 tests/data/wiki.test.raw > $O/ppl.txt
EX=$(cygpath -w $O/excerpt.txt); PP=$(cygpath -w $O/ppl.txt)
run() {  # name args...
    local name=$1; shift
    "$CB" "$@" > $O/cb.$name.out 2> $O/cb.$name.err < /dev/null; echo $? > $O/cb.$name.rc
    "$CF" "$@" > $O/cf.$name.out 2> $O/cf.$name.err < /dev/null; echo $? > $O/cf.$name.rc
    local a b
    a=$(grep -v -E "^(pp|tg): " $O/cb.$name.out | sha256sum | cut -c1-16)
    b=$(grep -v -E "^(pp|tg): " $O/cf.$name.out | sha256sum | cut -c1-16)
    echo "$name rc $(cat $O/cb.$name.rc),$(cat $O/cf.$name.rc) $a $b $([ "$a" = "$b" ] && echo same || echo DIFF) $(wc -c < $O/cf.$name.out)B"
}
suite() {  # key chunks-per-token model flags...
    local k=$1 c=$2 m=$3; shift 3
    run $k.generate generate "$(cygpath -w $m)" "The capital of France is" -n 64 --temp 0 "$@"
    run $k.logits logits "$(cygpath -w $m)" --file "$EX" --top 20 "$@"
    run $k.ppl perplexity "$(cygpath -w $m)" --file "$PP" --ctx-size 128 --chunks 4 "$@"
    run $k.ppl_tok perplexity "$(cygpath -w $m)" --file "$PP" --ctx-size 128 --chunks $c --per-token "$@"
}
date
suite 06q8 4 $M06 --device cpu --threads 16
suite 06q4 4 $M06Q4 --device cpu --threads 16
suite 06q5 4 $M06Q5 --device cpu --threads 16
suite 06q8f32 2 $M06 --device cpu --threads 16 --cache-type-k f32 --cache-type-v f32
run 06q8.t1.generate generate "$(cygpath -w $M06)" "The capital of France is" -n 32 --temp 0 --device cpu --threads 1
run 06q5.t1.logits logits "$(cygpath -w $M06Q5)" --file "$EX" --top 20 --device cpu --threads 1
suite 30 1 $M30 --device cpu --threads 16
run 30.t12.generate generate "$(cygpath -w $M30)" "The capital of France is" -n 32 --temp 0 --device cpu --threads 12
date
