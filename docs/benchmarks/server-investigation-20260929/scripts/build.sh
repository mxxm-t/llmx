#!/bin/bash
# build.sh TREE [TARGETS...]: the worktree TREE under /opt/claude-work/llmx-p2-perf2 built with Vulkan (Release) in b/, in container llmx-p2-perf2-b-TREE on CPUS (default 4-7,12-15).
# Writes TREE/build.done or TREE/build.fail, and keeps the configure and compile logs in TREE/b.cfg.log and TREE/b.build.log.
W=/opt/claude-work/llmx-p2-perf2
tree=$1; shift
targets="$*"
T=$W/$tree
CPUS=${CPUS:-4-7,12-15}
rm -f $T/build.done $T/build.fail
tg=""; for t in $targets; do tg="$tg --target $t"; done
cmd="set -e; ulimit -c 0; cmake -S . -B b -DCMAKE_BUILD_TYPE=Release -DLLMX_HAS_BACKEND_VULKAN=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > b.cfg.log 2>&1; cmake --build b --parallel 8 $tg > b.build.log 2>&1"
if timeout 5400 docker run --rm --init --name llmx-p2-perf2-b-$tree --cpuset-cpus $CPUS --ulimit core=0 -v /opt/claude-work/ccache:/ccache -e CCACHE_DIR=/ccache -v $W:$W -w $T llmx-p2-dev:vulkan-ccache bash -c "$cmd"; then touch $T/build.done; else touch $T/build.fail; docker stop -t 5 llmx-p2-perf2-b-$tree >/dev/null 2>&1; fi
find $T -maxdepth 4 -type f \( -name core -o -name "core.[0-9]*" \) -delete 2>/dev/null || true
