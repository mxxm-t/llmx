#!/bin/bash
# gate_ident.sh DEV OUT: byte identity of mainref and gate CLI outputs (generate, logits, per-token perplexity) on DEV for the Qwen3 files, run inside the build image.
A=/opt/claude-work/llmx-p2-kqcols/mainref/b/llmx
B=/opt/claude-work/llmx-p2-kqcols/gate/b/llmx
dev=$1; O=$2; mkdir -p $O
T=/opt/claude-work/llmx-p2-kqcols/gate/tests/data/wiki.test.raw
head -c 4000 $T > $O/excerpt.txt
P="The history of the printing press begins in"
M06=/m/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b
M8=/m/models--lmstudio-community--Qwen3-8B-GGUF/snapshots/07ebe812301319d9947477e3a94ab8aa587bb3af
run() { local name=$1; shift
  for arm in A B; do exe=${!arm}; "$exe" "$@" 2>/dev/null | grep -vE "^(pp|tg):|tok/s" > $O/$name.$arm; echo "rc=${PIPESTATUS[0]}" >> $O/$name.$arm; done
  if cmp -s $O/$name.A $O/$name.B && grep -q "rc=0" $O/$name.A; then echo "SAME $name $(sha256sum < $O/$name.A | cut -c1-16)" >> $O/summary.txt; else echo "DIFF $name" >> $O/summary.txt; fi; }
sha256sum $A $B > $O/summary.txt; $A --version >> $O/summary.txt; $B --version >> $O/summary.txt
for m in "q4km06:$M06/Qwen3-0.6B-Q4_K_M.gguf" "q4006:$M06/Qwen3-0.6B-Q4_0.gguf" "q5km06:$M06/Qwen3-0.6B-Q5_K_M.gguf" "q4km8:$M8/Qwen3-8B-Q4_K_M.gguf" "q6k8:/m2/Qwen3-8B-Q6_K.gguf" "q408:/m2/Qwen3-8B-Q4_0.gguf"; do
  n=${m%%:*}; f=${m#*:}
  if [ $dev = cpu ] && [ ${n: -1} = 8 ]; then run gen-$n generate $f "$P" -n 32 --temp 0 --device $dev; continue; fi
  run gen-$n generate $f "$P" -n 64 --temp 0 --device $dev
  run logits-$n logits $f "$(head -c 1500 $O/excerpt.txt)" --last 3 --device $dev
  run pplt-$n perplexity $f --file $O/excerpt.txt -c 128 --per-token --device $dev
done
echo DONE >> $O/summary.txt
