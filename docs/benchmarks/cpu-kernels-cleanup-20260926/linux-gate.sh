#!/bin/bash
# Inside the gate container, --cpus 6 with no GPU device: the branch tree mounted at /cf against main at /cb on the CPU.
set -u
ulimit -c 0
mkdir -p /root/.cache/huggingface; ln -sfn /m /root/.cache/huggingface/hub
O=/o
date; cat /proc/loadavg
echo "== builds"
for T in cb cf; do
    cd /$T
    cmake -S . -B b-cpu -DCMAKE_BUILD_TYPE=Release -DLLMX_HAS_BACKEND_VULKAN=OFF -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > $O/$T.configure.log 2>&1 || { echo "$T configure failed"; exit 1; }
    cmake --build b-cpu --parallel 6 > $O/$T.build.log 2>&1
    echo "$T build rc $?, warnings $(grep -c 'warning:' $O/$T.build.log), errors $(grep -c 'error:' $O/$T.build.log)"
    echo "/$T $(git log --oneline -1 | cut -c1-60) dirty $(git status --short | grep -v '^??' | wc -l) src $(git rev-parse HEAD:src | cut -c1-12) $(grep CMAKE_HOME_DIRECTORY b-cpu/CMakeCache.txt) $(sha256sum b-cpu/llmx | cut -c1-16) $(b-cpu/llmx --version)"
done

echo "== kernel harness: every matmul, group, expert and gather output hashed at 1, 6 and 16 threads, decode dots on and off"
sed -e 's/for (int threads : {1, 6})/for (int threads : {1, 6, 16})/' /t/ck_ident.cpp > /tmp/ck_ident3.cpp
grep -c "{1, 6, 16}" /tmp/ck_ident3.cpp
for T in cb cf; do
    cd /$T
    g++ -std=c++17 -O3 -DNDEBUG -mavx2 -mfma -mf16c -Wall -Wextra -Isrc -Ib-cpu/generated /tmp/ck_ident3.cpp -o $O/$T.ident -pthread > $O/$T.ident.build.log 2>&1
    echo "$T harness build rc $?, warnings $(grep -c 'warning:' $O/$T.ident.build.log)"
    s=$(date +%s)
    $O/$T.ident > $O/$T.ident.out 2> $O/$T.ident.err
    echo "$T harness rc $?, $(cat $O/$T.ident.err), $(( $(date +%s) - s )) s, threw $(grep -c threw $O/$T.ident.out)"
done
if cmp -s $O/cb.ident.out $O/cf.ident.out; then echo "harness IDENTICAL: $(wc -l < $O/cf.ident.out) lines"; else echo "harness DIFFERS"; diff $O/cb.ident.out $O/cf.ident.out | head -10; fi
grep gather $O/cf.ident.out

echo "== build-time requirement: a unit including the CPU backend, compiled without the ISA flags"
printf '#include "backends/cpu/cpu_backend.hpp"\nint main() { return 0; }\n' > /tmp/req.cpp
for flags in "" "-mavx2 -mfma" "-mavx2 -mfma -mf16c" "-march=haswell"; do
    cd /cf
    g++ -std=c++17 -O2 -c $flags -Isrc -Ib-cpu/generated /tmp/req.cpp -o /dev/null > /tmp/req.log 2>&1
    echo "cf flags [$flags]: rc $?, error lines $(grep -c 'error' /tmp/req.log)"
    grep -m1 "error" /tmp/req.log | cut -c1-200
done

echo "== CTest on the branch"
timeout 3600 ctest --test-dir /cf/b-cpu --output-on-failure > $O/cf.ctest.log 2>&1
echo "ctest rc $?"
grep -E "tests passed|tests failed|Failed|\*\*\*" $O/cf.ctest.log | head -10

echo "== CLI identity on the CPU, name rc(main,branch) hash(main) hash(branch)"
M06=/m/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf
U=/m/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b
M06Q4=$U/Qwen3-0.6B-Q4_0.gguf
M06Q5=$U/Qwen3-0.6B-Q5_K_M.gguf
M8=/m/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf
M30=/c/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf
ls -la $M06 $M06Q4 $M06Q5 $M8 $M30 | awk '{print $5, $NF}'
mkdir -p $O/id
head -c 1500 /cf/tests/data/wiki.test.raw > /tmp/excerpt.txt
head -c 4000 /cf/tests/data/wiki.test.raw > /tmp/ppl.txt
run() {  # name args...
    local name=$1; shift
    for arm in cb cf; do
        timeout 3600 /$arm/b-cpu/llmx "$@" > $O/id/$arm.$name.out 2> $O/id/$arm.$name.err < /dev/null
        echo $? > $O/id/$arm.$name.rc
    done
    local a b
    a=$(grep -v -E "^(pp|tg): " $O/id/cb.$name.out | sha256sum | cut -c1-16)
    b=$(grep -v -E "^(pp|tg): " $O/id/cf.$name.out | sha256sum | cut -c1-16)
    echo "$name rc $(cat $O/id/cb.$name.rc),$(cat $O/id/cf.$name.rc) $a $b $([ "$a" = "$b" ] && echo same || echo DIFF) $(wc -c < $O/id/cf.$name.out)B"
}
suite() {  # key chunks-per-token model flags...
    local k=$1 c=$2 m=$3; shift 3
    run $k.generate generate $m "The capital of France is" -n 64 --temp 0 "$@"
    run $k.logits logits $m --file /tmp/excerpt.txt --top 20 "$@"
    run $k.ppl perplexity $m --file /tmp/ppl.txt --ctx-size 128 --chunks 4 "$@"
    run $k.ppl_tok perplexity $m --file /tmp/ppl.txt --ctx-size 128 --chunks $c --per-token "$@"
}
suite 06q8 4 $M06 --device cpu --threads 6
suite 06q4 4 $M06Q4 --device cpu --threads 6
suite 06q5 4 $M06Q5 --device cpu --threads 6
suite 06q8f32 2 $M06 --device cpu --threads 6 --cache-type-k f32 --cache-type-v f32
run 06q8.t1.generate generate $M06 "The capital of France is" -n 32 --temp 0 --device cpu --threads 1
run 06q5.t1.logits logits $M06Q5 --file /tmp/excerpt.txt --top 20 --device cpu --threads 1
suite 8q8 1 $M8 --device cpu --threads 6
suite 30 1 $M30 --device cpu --threads 6
run 30.t16.generate generate $M30 "The capital of France is" -n 32 --temp 0 --device cpu --threads 16
date; cat /proc/loadavg

echo "== Python suite on the branch, CPU, every component but the server"
cd /cf
s=$(date +%s)
timeout 7200 python3 tests/run_tests.py --exe b-cpu/llmx --no-perf-floor --require-baseline \
    --only version,cli,reference-generator,reference-consumer,roundtrip,raw-blocks,perf,tokenizer,perplexity,f32,moe,split,shards,server-load,chat,threads,baseline > $O/suite.log 2>&1
echo "suite rc $?, $(( $(date +%s) - s )) s"
grep -E "^  [a-z0-9-]+ +(PASS|FAIL|SKIP)" $O/suite.log | tail -40
date; cat /proc/loadavg
echo "== the suite's server component on the branch, servers at --threads 6"
python3 /t/srv_inner.py /cf > $O/srv.cf.log 2>&1
tail -1 $O/srv.cf.log
if ! tail -1 $O/srv.cf.log | grep -q PASS; then
    echo "== the server component on main, servers at --threads 6"
    python3 /t/srv_inner.py /cb > $O/srv.cb.log 2>&1
    tail -1 $O/srv.cb.log
fi
date; cat /proc/loadavg
