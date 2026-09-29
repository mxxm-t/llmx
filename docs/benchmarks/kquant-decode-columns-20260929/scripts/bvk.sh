#!/bin/bash
# bvk.sh TAG RENDER TREE ENVS [ARGS]: backend-vulkan test of TREE on one card with VAR=VALUE,... in the environment, output in jobs/TAG.txt.
W=/opt/claude-work/llmx-p2-kqcols
tag=$1; RENDER=$2; tree=$3; envs=$4; shift 4
mkdir -p $W/jobs
GROUPS_="--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)"
envx=""; for e in ${envs//,/ }; do envx="$envx $e"; done
timeout 3600 docker run --rm --init --name llmx-p2-kqcols-t-$tag --cpuset-cpus ${CPUS:-4-7,12-15} --ulimit core=0 --device /dev/dri/renderD$RENDER $GROUPS_ -v $W:$W -w $W/$tree/b llmx-p2-dev:vulkan-ccache bash -c "ulimit -c 0; sha256sum ./llmx-backend-vulkan-test; env $envx ./llmx-backend-vulkan-test $*; echo exit=\$?" > $W/jobs/$tag.txt 2>&1 || docker stop -t 5 llmx-p2-kqcols-t-$tag > /dev/null 2>&1
find $W -maxdepth 4 -type f \( -name core -o -name 'core.[0-9]*' \) -delete 2>/dev/null
