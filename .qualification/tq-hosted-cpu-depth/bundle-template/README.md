# TQ clean-build depth-capture preparation v2

Local preparation only. No v2 files were transferred to the rig, and no inference or capture was run. The probes were compiled in the separately pinned hosted job described below. The candidate binds to audited hosted matched
build run 40101812090: clean commit a0fa25492b3ec9badaa43f8cd489fbb68fe06494,
Git tree 2f66f4090866e2a0733d504f7d3526540a63f020, CLI
abb515f5aa00189932b4777b69c61d90024e80770152d0473bd77577808437f8, Vulkan
library 946cef2e728500ec7f629c6773c75fc82b7056abdd65d691eaa7a6705845050e,
config header cffa6da5cbbeb9d258e8ed6d4505d16a512511ae681cc798d3afb30dea978749,
and build-info header de82d9d8624a24c9060116e84d07a2546497cd1ff90cf65cc998415c448eb1b0.
The plan pins the receipt, CMake cache, compile commands, four generated TQ
row/tile dot modules, image ID and hosted archive digests. The receipt covers
850 inventoried build outputs.

The default and explicit-F32 probes were compiled in hosted run 40106666694
against that exact library and image. The verified hosted artifact is
12564704111 (official SHA256
`0fa344713bf84d30bc5a78fab86a305be3e7d139258ddb93311f40183835448e`). The
probe-build receipt SHA256 is
`1559727f04938dc98975a85ae16ef260753ab61c497da8a3fbf46b9a64e2077c`; the
default and F32 executable hashes are
`2e5198b08e9064dcf3e33ecf8ce47d7f2f2b84d24cfa86a6f6c04225e3df045b` and
`6861305e727d287526793edcc42e6ea16c594cc890ca5aca01961b96b01d2a84`. The
receipt ties both source hashes and binaries to the candidate library. This is
compile provenance only; neither probe has been run on a model or GPU. The
local binding validator rechecks those receipts, hashes and workflow source
files before any v2 staging.

The capture payload checks runtime/input hashes before and after, verifies
tokenizer and version/path witnesses, and requires 512 scored positions
(16383 through 16894) with 50 finite logits per position. It does not compare
against HF. Scope is 16 cells: two released TQ GGUFs, CPU and MI50 Vulkan,
auto/F16 and explicit F32, raw and chat. Comparison remains with
`tests/common.py:hf_depth_top1` at the unchanged 0.1 margin. This capture adds
no full-array or NLL gate. Independent HF per-cell outputs and execution
receipts are still required before comparison.

Remote layout is
`/zpool1/llmx-xdev-quant-20261020/tq-pair-20261024/clean-depth-capture-v2`:
`source`, `candidate-build`, `build-receipt.json`, `image/build-image.tar.gz`,
`probes`, `inputs`, `fixtures`, `models` and `captures`. The payload commands
are exactly:

    python3 -B /bundle/build-probes.py
    python3 -B /bundle/capture-one.py <tq1-0|tq2-0> <default|f32> <raw|chat> <cpu|0>

`run-capture.py` is the outer adapter. It verifies the hosted build receipt
and all 850 build-artifact hashes, source HEAD/tree/clean state, image ID,
model/input/fixture hashes and probe receipt before capture. It reuses
`resource-owner.py` from the existing TQ F16KV reference runner for host/disk/
CPU admission, peer reservations, cgroup monitoring, live-floor checks and
container identity inspection. For Vulkan it also calls the existing timing
owner in `gpu-owner.py` for render-node/PCI/VRAM/fuser/device-container checks.
The cleanup path removes only a verified exact ID/name/image/label and records
failure receipts. Both owner files are copied byte-for-byte and hash-pinned.

Run the pure local checks with:

    python validate-binding.py
    python capture-contract-smoke.py
    python run-capture.py probe-build --self-test

A real capture command requires `--execute`, `--resources-released`, a fresh
`--admission` JSON and a unique `--run-id`. Do not invoke `probe-build`: v2
uses the already-built, hash-pinned binaries. The admission file supplies a
current contiguous CPU set/quota/container name and, for Vulkan, a freshly
checked device record. Its keys are `cpu_set`, numeric `cpu_quota`, integer
`threads`, `container_name`, and `device` (null for CPU/probe build). For
example, `device` contains the just-checked `node`, `pci`, `uuid`, `driver` and
`name`; the runner independently rechecks render-node PCI, busy/VRAM, power,
fuser and container use. The outer command forms are:

    python3 -B run-capture.py capture --execute --resources-released --admission <fresh-admission.json> --cell cpu-tq1-0-auto-f16-raw --run-id <unique-id>

The fixed capture resource policy is the previously used 10 GiB
container cap, 14 GiB initial host-availability threshold, 4 GiB minimum live
reserve plus unused cap, 5 GiB disk floor, 256 PID limit, no network, read-only
inputs and a 2100-second payload timeout. The resource owner adds unused peer
reservations to admission/live-floor checks. No CPU placement or GPU
availability is considered current until fresh admission passes.

This isolated retry workflow selects only the terminally failed TQ1_0/F32 CPU
raw and chat cells. The prior 6 GiB cap run is retained; Docker reported
OOMKilled for both cells, with a cgroup oom_kill event for chat. The retry uses
the same probe, model, fixtures, full-context workload, child timeout and
numerical checks, with the already specified 10 GiB cap and 14 GiB admission.
The workflow matrix excludes all other models and policies.

Before capture, stage exact source revision a0fa25492b3ec9badaa43f8cd489fbb68fe06494 with Git tree 2f66f4090866e2a0733d504f7d3526540a63f020 at `/source`, load and inspect
image `sha256:e369e69469f20d3940d3689abdbcc33545a5db0361807119ef003b1beb759d85`,
and stage the pinned TQ models at `/models`. Extract `probe-build/probes/default-probe`, `probe-build/probes/f32-probe`, and `probe-build/probes/probe-build.json` from the verified hosted artifact and stage those exact files flat at the pinned v2 probe path. The current plan pins the
existing model directory `/zpool1/llmx-xdev-quant-20261020/tq-pair-20261024/models`.
Stage frozen inputs and fixtures at the plan paths and verify the complete
manifest and each listed artifact. Do not invoke probe-build mode for v2: the exact compiled probes are already pinned. Capture execution remains pending root review and fresh admission.

The audited build receipt records passed, container attach exit 0, container
exit 0 without OOM, exact source HEAD and clean source status. That is build
provenance only, not a depth, correctness or performance result.
