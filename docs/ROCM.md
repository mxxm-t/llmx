# ROCm backend (planned)

Research and proposed delivery plan, 2026-10-09, based on main a1b8211b4.
No ROCm backend is implemented or qualified by this document. The source
comparison and pinned references are in [the research record](research/rocm-20261009.md).
The user requested the plan and on 2026-10-09 authorized its bounded hardware
probe before Vulkan tensor split is complete. The remaining implementation
follows the existing order in [TENSOR-SPLIT](TENSOR-SPLIT.md). The closed
patched-kernel/driver route stays closed.
This combines two research proposals, dated 2026-10-08 and 2026-10-09,
reviewed on 2026-10-09. The research record states the accepted inputs and the
corrections. The launch correctness smoke passes on one MI50, and guarded peer
writes followed by host-synchronized ordered sums pass on groups of two, three
and four MI50s in the three public allocation modes. The subsequent one-block
device-signal probe passes with uncached payloads at all three group widths;
ordinary and fine-grained payloads fail that probe. A subsequent bounded
multi-block eager/graph screen passes with uncached storage at those widths,
including missing-member and stale-epoch controls. Model graph integration,
general failure recovery and qualified performance remain open. Authorization
for the probe does not authorize landing an incomplete backend.

## Outcome and scope

Implement the existing `Backend` and `Collective` contracts using HIP, with
llmx's own kernels and ownership rules. Reuse the loader, model architectures,
placement, pass scheduler, server, dtype resolution and reference tests.
There is no second model runtime, graph scheduler or precision system.

The first delivery target is Linux MI50/gfx906. The completion target is all
currently supported model operations, quant types and requested dtypes, including
hybrid layers, routed experts, embedded drafts, layer splits and tensor groups.
The dense Qwen3 Q8_0 milestone is a bounded first implementation, not a claim
that the backend is complete. Unsupported operations/types are refused at load
through existing capability checks, before uploads or execution.

Strix Halo/gfx1151 is an explicit future hardware target. The user confirmed on
2026-10-09 that no machine is available. Until then, avoid shared wave64
assumptions and compile wave32 variants of kernels already being implemented.
Device-specific memory accounting, profile rows and other seams wait for
hardware, as do numerical, performance and support qualification. Newer CDNA and RDNA devices use
the same qualification procedure. Linux is the planned platform; Windows ROCm
support would be a separate scope change even if a vendor SDK supports a device.

GPU SDK/runtime dependencies are the existing backend exception. HIP is the
foundation. Evaluate RCCL early as an optional ROCm-backend dependency for large
collectives, pipeline transfers and fallback; excluding it in advance would
commit us to rebuilding transport before measuring the existing implementation.
The user authorized RCCL behind its own off-by-default build option on
2026-10-09; enabling a path still requires the hardware and numerical checks
below. A single-GPU backend, including future Strix Halo, needs no RCCL.
The conservative HIP path and custom sum must remain usable without RCCL.
This plan adds no runtime dependency. rocBLAS, AITER, PyTorch and Transformer
Engine remain research inputs for this delivery; no external source is copied
into llmx.

## What the research changes

| Source | Take into this design | Keep outside this delivery |
| --- | --- | --- |
| mx fork | Peer-write F32 collectives, correctly ordered signals, measured one/two-phase selection, concurrent rank launch, stable decode graphs, packed quant layouts | Its environment-variable control surface, GGML model plumbing, silent BF16 wire reduction, hardware assumptions extended without proof |
| vLLM | Qualify topology and pointers before using a custom collective; stable storage for graph replay; explicit fallback; separate batch invariance from fast algorithms | Its newer-Instinct whitelist as evidence for MI50, framework dependencies, lossy communication by default |
| SGLang/AITER | Separate deterministic sums from alternative reduction orders; bound captured shapes and memory; preserve logical rows under padding | Its complete distributed runtime, AITER dependency, optional quantized communication |
| Megatron Core | Keep expansion projections local and reduce after contraction; overlap only independent work; respect data dependencies | Training gradients, optimizer communication, sequence/expert parallelism added without a workload |
| llmx Vulkan | Existing buffer/ticket/KV contracts, quant ownership, dispatch witnesses, tests, measured device-profile structure | RADV-specific constants, descriptor rules, shader names or wave64 assumptions presented as portable HIP tuning |

The performance hypothesis is specific: HIP can remove the per-sum host
submission/sync-file cost seen in Vulkan and can use suitable native dot/matrix
instructions. It is not a promise that translating GLSL makes every kernel faster.
The first probe must establish the communication gain before full backend work.

## Owners and boundaries

| Concern | Owner and boundary |
| --- | --- |
| HIP loading, devices, allocation, queues, tickets, errors, kernel dispatch and graph lifetime | One ROCm backend module below `model/`; public factory follows the Vulkan backend's shape |
| Peer buffers, membership, sum algorithm, launch coordination and communication scratch | ROCm implementation of `Collective`; no new server/model collective selection |
| Weight storage metadata and decoding definitions | Existing `quant/` owner; one ROCm descriptor per supported weight type selects its kernel family |
| Execution dtype and fallback | Existing model placement resolver; backend advertises only qualified implementations and reports executed paths |
| Physical KV/state layout and copies | ROCm backend; logical sequences, slots and transactions remain in the existing model layer |
| File loading, direct reads and progress | Existing inference loader; HIP only implements its buffer adoption/write contract |
| Device facts and tuning | Existing device-profile concern, with API-specific measured rows and narrowly owned HIP instruction traits |
| Placement and shared-memory accounting | Existing model fit consumes backend facts; the CLI does not calculate a second budget |
| Builds and packaging | CMake/config owners; add the ROCm option only with the implementation, and update both generated and fallback config |

The backend is compiled only with `LLMX_HAS_BACKEND_ROCM`, off by default, like
`LLMX_HAS_BACKEND_VULKAN`. Add it with the implementation to CMake,
`cmake/llmx-config.hpp.in` and the fallback `src/config.hpp`; without it,
`--device rocm:N` reports that this build lacks the backend. RCCL has a separate
build option under ROCm, also off by default. Enabling RCCL
without the ROCm backend is a configuration error. Neither option is a runtime
environment variable.

Keep the public header small and implementation below it. Split an internal
collective, kernel family or runtime loader into another file when it has an
actual owner and caller, not to create a speculative backend framework. Reuse
the existing CPU/device test oracles and test parameterization rather than
forking their entire suites. Match the project's exception and teardown rules:
user input, unsupported hardware and allocation errors are runtime errors;
assertions express internal invariants. Deferred failures keep their first cause,
drain owned work where possible, and never free queued destinations early.

## Hardware configuration, including Strix Halo

`src/backends/device_profile.hpp` already separates `DeviceCaps` from
`DeviceProfile`. HIP should fill facts from its own API. Vulkan does not need to
be installed or queried to configure HIP. The Vulkan tables supply hypotheses
for sweeps, not production HIP defaults.

When HIP becomes the second consumer, make profile selection distinguish the
backend API, device ISA and compiler/runtime identity. Preserve every current
Vulkan choice. Share only fields/helpers with the same meaning for both callers;
Vulkan-specific bit flags such as its Q8 reduction forms remain with that
implementation. A tuning record names the kernel family/revision it measured.
An unknown device gets a supported conservative implementation and an unqualified
performance status, not a guessed MI50 profile.

| Category | How to use it |
| --- | --- |
| Reported facts | Query wave width, compute units, LDS/workgroup limits, allocation limits, integrated-memory status and pairwise peer access; retain device/driver/runtime/ISA identities |
| Compiler/ISA facts | Keep instruction availability in one HIP traits owner, checked against built code objects and compiler features; HIP does not expose every useful instruction as a runtime boolean |
| Qualified policy | Advertise F16/BF16/int8 only after reached kernel families pass their numerical contract; an instruction or conversion intrinsic alone is insufficient |
| Measured choices | Tile sizes, row widths, splits, packing, attention crossover, graph shapes and collective crossover live in one backend tuning owner |
| Kernel invariants | Quant block layout and a kernel's required lane grouping stay beside that kernel; they are not tuning knobs |

Compile explicit code objects for selected targets; do not spoof a device's ISA.
The MI50 target is gfx906, not a generic gfx9 binary assumed to expose every
dot instruction. Confirm the intended instructions in the pinned compiler's
output; a generic target's documentation is not an executed-path witness.
Wave reductions use the target's actual wave width and masks wide enough for it.
MI50 uses wave64 and has no CDNA MFMA path. RDNA3.5 has a wave32 WMMA path with
different register/tile behavior. A matrix-instruction family is added beside
the ordinary dot family when its implementation exists, with one dispatch site.
Do not scatter checks for gfx906 or gfx1151 through models or the CLI. Compile
wave32 and wave64 variants of portable kernels being implemented; inspect ISA and resource
counts on each claimed target. Compilation is not hardware qualification.

Strix Halo shares physical RAM with its CPU. Its later hardware bring-up must:

- Account for GPU-addressable limits and host/cgroup headroom without adding
  CPU and GPU free-memory figures as if they were independent pools. Report
  host-resident copies and shared allocations accurately to the existing fit.
- Compare ordinary device allocation/upload with supported host-page registration
  and direct GPU reads. Integrated memory alone proves neither zero-copy support
  nor coherence nor speed. Pin/registration lifetime follows the Buffer owner.
- Measure cold/warm load, memory pressure, KV growth, follow-up restore and
  concurrent CPU traffic. Avoid retaining both mapped and adopted copies unless
  needed; preserve the loader's existing `reads_in_place` contract.
- Use a pinned supported OS/kernel/ROCm combination. Record firmware and memory
  mapping limits; llmx does not alter firmware, sysctls or driver settings.
- Benchmark mx on that same machine. An MI50 result says nothing about Strix Halo
  throughput. Strix is initially a single-GPU target, not an MI50 PCIe group.

Native BF16 or WMMA must not silently round exact decoded weights to fit a
matrix instruction. [PRECISION](PRECISION.md) defines activation rounding,
retained F32 operations and frozen error budgets. Use a conforming decomposition
or wider fallback until the faster form is qualified. Weight format support
remains separate from dtype support.

## Compute kernels and packed weights

Develop a small set of families in their existing operation boundaries:

| Family | First implementation and measured optimization |
| --- | --- |
| Quantized decode | Coalesced packed-weight rows, reuse weights across generated columns, and qualified integer dot forms; keep logical row classes independent of physical batch size |
| Quantized prefill | Tiled products with bounded shared/register storage; compare unpack-in-tile with exact packed layouts and activation preparation costs included |
| F32/BF16 policy | Original or explicitly rounded activations with F32 sums, including dense, grouped, routed and head calls; no weight-type-based precision shortcut |
| Attention | Decode and prompt kernels over the same paged KV owner; online softmax and F32 sums, tails, GQA and both cache-side types; matrix instructions only under the declared arithmetic policy |
| Routed experts | Existing routing and grouped projection API; tile by actual expert work without inventing a second router or expert-parallel scheduler |
| Hybrid/state and drafts | Port the existing convolution, delta rule, gated norm and draft operations with checkpoint/rollback behavior first; optimize only after their recurrent numerical checks |

One backend weight descriptor selects implementations for every caller. Keep
canonical quant decoding in its existing owner. An internal repack is an exact
layout change, not a new file format or precision policy. It must support the
loader's partial writes and tensor shards without repacking incomplete data.
Choose either completion at an existing valid boundary or a deferred pack before
first use, with the stream ordering proved. Do not add model names to the loader
or dispatch to select a faster kernel.

Count packed bytes and temporary scratch in `resident_bytes` and the existing
fit queries; release upload/packing scratch when no outstanding ticket needs it.
Cache keys include allocation generation, byte span, storage type, dimensions,
strides and the packing variant. Activation caches also include their block
layout and executed dtype. The review's same-flat-buffer/different-width int8
case becomes a regression for both backends, not a HIP copy of the defect.

MI50's dot kernels, CDNA's MFMA and RDNA's WMMA are distinct implementations
behind one operation dispatch. Start with a working portable form, then add the
instruction family that an actual target can run. Measure loads, packing,
register spills, occupancy and launch gaps on real shapes before changing tile
constants. Fused epilogues use existing grouped/residual operations and must
preserve their ordering and alias rules. Newer-device support is not achieved by
expanding an architecture whitelist around the MI50 kernel alone.

## Execution and graph strategy

Begin with a correct ordered HIP stream, ticketed events, bounded reusable
workspaces and the existing pass boundaries. Compile kernels ahead of time.
Follow Vulkan's optional-runtime behavior: a CPU invocation must still start
without HIP installed. The proposed implementation uses ordinary C++ HIP API
loading and embedded per-target code objects loaded through the HIP module API;
the first build milestone proves this on the pinned toolchains. No runtime
compiler, Python package or model-time kernel download is required.

Add decode graph replay before calling the multi-GPU backend performance-ready.
The backend owns HIP graph objects and stable argument/KV-view buffers; the
model's pass/stage API remains the source of work. Use existing boundaries where
possible. If group capture demonstrably needs a boundary the interface lacks,
add one small backend-neutral seam with CPU/Vulkan behavior and tests, not a
second scheduler. This is an explicit design check in milestone 4.

Replay must validate the operation sequence, kernel/dtype/row-class choices,
shape/strides, group membership and buffer allocation generations. Dynamic token
positions, KV block lists and routing metadata live in stable device descriptors
updated with proper stream dependencies. Pointer reuse alone is not a cache key.
Bound graph variants and their memory; allocation growth, weight repacking,
model replacement and context teardown invalidate dependent graphs safely.
Capture cannot commit sequence state, consume random samples or publish logits.
Warmup must preserve the caller's state. Eager and replay paths give the same
bits for the same policy and row classes, including padded and mixed batches.

One member's graph must not wait on another member that the host cannot launch
until the first finishes. Prepare every member before any launch and use the
existing recorder/collective ownership to avoid duplicate worker pools. Measure
rank start skew and CPU cost with graphs on and off. Ordinary scheduling and
resource limits still apply when other users occupy the GPUs or CPU cores.

## Collective design and proof

Implement a conservative F32 sum first: ordered copies plus an ordered reduction
with explicit queue dependencies. It is the correctness fallback and baseline,
not the fast-path performance claim. Probe every directed peer pair; PCIe-root
labels are descriptive, not sufficient proof of visibility or performance.

The fast candidate follows the measured mx protocol: peer writes into owned
staging, correctly scoped release/acquire signals, and the local contribution
read from its original buffer. Use fixed member-order F32 accumulation on every
member. Start with a one-phase small-message kernel, then a two-phase
reduce-scatter/gather for larger messages only if comparison with RCCL justifies
implementing it. Custom paths preserve the chosen reduction order; padding and
tails are exact. The threshold is measured across message sizes and group widths,
not copied from mx.

The proposed hybrid uses custom AR where small decode messages benefit and RCCL
where large messages or transport benefit. Compare F32 RCCL all-reduce and
byte-preserving send/receive against the HIP baseline at two, three and four
members, including stage boundaries. Qualify the pinned RCCL/runtime on gfx906,
graph capture/replay, asynchronous failure handling and communicator teardown.
RCCL stays behind the existing ROCm collective/transfer owners; model, loader and
server callers acquire no vendor-specific logic. Record its package, memory and
startup costs before deciding whether to adopt it.

F32 RCCL does not imply a fixed member-order sum. Test repeated runs, equivalent
row classes, message-size transitions and graphs against llmx's numerical
contract; a path that fails it is ineligible. Byte transfers perform no reduction
and need their own visibility/lifetime tests. Do not use BF16 conversion to make
RCCL pass a performance screen. A general multi-node transport remains outside
this delivery even though RCCL can provide one later.

The 2026-10-09 numerical screen in [STATUS](STATUS.md) confirms this distinction:
RCCL send/receive followed by ordered local sums passes at two, three and four
MI50s, while native F32 all-reduce differs from member order at three and four.
Those native configurations cannot be the ordered fallback. Retain the transfer
candidate and prove any sum configuration separately; a two-member pass on the
small screen does not qualify all shapes or graph execution.

The subsequent device-signal screen also narrows the memory candidate: separate
uncached signals and uncached payloads pass 512 changing epochs at each of two,
three and four MI50s, without the fine-grain environment override. The same
compiled protocol reads stale payloads with ordinary and fine-grained payload
allocations, despite completed signal handshakes. Retain those failures and
use uncached staging for the next probe. This is one block per member and a
finite set of inputs, not qualification of arbitrary peer access, graph replay
or the complete collective. The compiled non-temporal hint does not establish
a cache-bypassing instruction: the recorded gfx906 payload loads are ordinary
global loads. See STATUS for the exact source, compiler and bounds.

Later multi-block eager/graph probes pass the bounded matrix in STATUS. The
lean candidate adds the ordered member sum to an in-place residual, matching
`Collective::sum_into`, and checks each chain's final result independently.
Its emitted scratch use is zero on gfx906 and gfx1151; the latter is compile
coverage only. The first matched Vulkan/HIP timing screen is recorded in STATUS;
its width-four graph runs show substantial host-launch skew in one block, so
the pooled latency is not a stable admission budget. The subsequent same-binary
wait-policy comparison gives consistent graph results when all ranks enqueue
before any host wait, while eager mixed chains still vary. Preserve the backend's
submission/completion separation, then complete fresh mx and conservative
transport controls and refresh the model budget. No production collective is
qualified by these screens.

Memory visibility is the first go/no-go proof. A peer-access query or successful
allocation is insufficient. Run repeated producer/consumer epochs with changing
sentinels, delayed members, cache reuse and the exact allocation kinds. Test
flags and payloads together. On MI50, investigate whether public allocation
flags alone provide the required behavior. mx's fast path relies on the external
HSA fine-grain PCIe setting on this platform. Vendor runtime and driver variables
are permitted deployment prerequisites (owner clarification, 2026-10-09). If the
pinned runtime needs this setting, document it in BUILD and OPERATING, set it
explicitly in the backend's Docker image and report whether the fast path is
available, including the reason when it is not. The conservative path remains
available and is reported. llmx's own runtime controls remain CLI flags;
temporary development variables may serve probes but are removed or converted
to flags before their code lands.

The collective owns signals, scratch, epochs and their outstanding tickets.
Prove scratch reuse across passes, signal wrap/reinitialization, startup skew,
graph capture/replay, cancellation, allocation failure and teardown. In-kernel
waiting requires a residency/progress argument and failure reporting: a member
that never launches must not leave an apparently healthy server waiting forever.
Test failures in isolated processes with bounded supervision; never reset shared
test-system devices to recover a test. Select fallback before launch. Do not attempt to
switch algorithms after some members have entered a barrier.

Keep actual wire precision in the evidence. mx can use BF16 RCCL reductions on
large messages; matching its int8 matrix inputs does not make those collectives
the same precision. That conversion is mx policy, not an RCCL requirement.
llmx keeps F32 sums/wire data under the current contract.
Record the reference's real path and retain the public best-path target; add an
F32-communication diagnostic control if supported. A lower-precision collective
would require its own approved policy and calibrated gate, not a hidden speed fix.

## Probe workloads and decision rule

The probe tests a concrete hypothesis before committing to a backend. Historical
Vulkan results identify likely gains in width-4 communication and prompt
attention, not a promise that changing APIs makes every kernel faster. The
2026-10-05 comparison in [STATUS](STATUS.md) already puts Qwen3-8B Q8_0 decode
at 77.9 tok/s on llmx int8 against 67.9 on mx ROCm on one MI50. A new backend
must also preserve llmx's strengths. Width 2 needs a profile outside the sum;
replacing a roughly 46 us sum with the reference's implied 42 us cannot explain
away its entire decode gap.

For Qwen3-32B Q8_0, 128 sums per generated token, a planning budget is
`(1 / reference_tok_s - 1 / no_sum_tok_s) * 1000000 / 128` microseconds.
The same-session 2026-10-06 record in STATUS gives:

| Group width | mx decode, tok/s, rounds 1/2 | llmx with sums removed, tok/s, rounds 1/2 | Derived budget per sum, us, rounds 1/2 |
| --- | --- | --- | --- |
| 4 | 49.8 / 48.9 | 61.6 / 62.1 | 30.1 / 34.0 |
| 2 | 33.3 / 33.9 | 36.0 / 36.3 | 17.6 / 15.2 |

These are estimates under unchanged non-collective work, not measured HIP
latencies or release gates. The no-sum binaries produce wrong logits, and their
data, layout and scheduling can affect other work. The earlier roughly 27 us width-4
estimate combines the older no-sum result with a later 50.8-51.0 tok/s reference;
it is a useful target, not a matched measured budget. The model comparison below
refreshes the controls in one environment; these original rows remain historical evidence.

| Probe | Required comparison |
| --- | --- |
| 500 dependent small kernels | Separate CPU enqueue time and device elapsed time; HIP eager and replay against a fresh Vulkan control. The historical 4.1 us dependent dispatch is context, not host launch time. Include concurrent member launch and full token-like chains. |
| Q8_0 decode matrix | Same 4096-wide matrix, output rows, activations, dtype and residency as Vulkan; report bandwidth, arithmetic witness and end-to-end time. A cache-resident microkernel is not model decode. |
| Collective | Widths 2, 3 and 4, including the 20 KiB row and a size sweep; ordered F32 oracle, custom AR, conservative HIP and RCCL candidates, with skew and graph replay. Width 3 uses a legal synthetic shape where a real model cannot split. |
| Prompt attention | Isolate QK and PV instruction throughput, then the complete attention op with the same heads, history, rows, softmax and accumulation policy. Historical 269/1050 ms values are the model's attention totals at 2048/4096 tokens; do not compare a standalone dot to those totals. |

Freeze thresholds and workload inputs before running. Advance the HIP design
when the token-like chain shows the qualified sum fits the refreshed width-4
budget and additional launch/compute work does not consume the saving. Report
width-2 and prompt findings independently. A failed eager-launch screen first
tests HIP graphs and launch coordination. If the combined result still cannot
meet its purpose, report that result and the measured alternatives before
expanding scope; a failed prototype does not justify a production backend stub.

HRX and direct HSA/KFD remain research alternatives, not automatic next steps.
The [HRX project](https://github.com/ROCm/hrx-system) identifies itself as early
access outside the official ROCm stack; its README does not qualify gfx906.
Its existence does not measure llmx launch overhead. Changing runtime or writing
packets below HIP needs a separate concrete proposal, dependencies and hardware
qualification. The closed kernel/driver-patch route stays closed.

The [2026-10-09 control matrix](benchmarks/rocm-controls-20261009/README.md)
now includes fresh direct mx calls and conservative HIP/RCCL transfers at
widths two, three and four. All bounded outputs pass; background CPU activity,
larger-vector losses and graph/eager differences remain in the record. The
[refreshed model budget](benchmarks/rocm-model-budget-20261010/README.md)
now measures current Vulkan main, its private one-row no-sum diagnostic and
shipped mx at widths two/four with F16 and int8 kept separate. All 20 processes
complete, with output/path witnesses and every activity flag retained. The
diagnostic's decode outputs are wrong by construction; its derived budget is
an estimate. The [Q8 compute probe](benchmarks/rocm-q8-compute-20261010/README.md)
now records both the initial HIP loss and a measured load-path improvement,
with all 64 matrix processes retained. The candidate remains slower than
Vulkan; profiling overhead and the ordinary-fixture limit stay explicit.
The follow-up isolates preparation at about 1.8 us and reduces F16 latency
with wave-uniform two-row ownership, while retaining int8 regressions and
the slower one-row and spilling occupancy controls. F16 remains 8-13 percent
slower than Vulkan in those cells. Full prompt attention is next; the residual
compute cost remains in the combined admission budget, not a waived gate.

## Delivery checkpoints

Each row is a feature branch with a concrete exit. Land no empty backend stub.
Unimplemented operations in the bounded first backend refuse by capability.
The current [AGENTS](../AGENTS.md) gates apply at each landing; mx comparisons
are phase-final gates, with short diagnostic screens during development.

| Step | Deliverable | Exit evidence |
| --- | --- | --- |
| 0. Contract and baseline | Freeze supported first models, matrix, runtime/compiler, source hashes, device mapping and profile observations; resolve relevant open review fixes at existing owners | Exact mx path/precision witnesses; current-main control; no unresolved unsafe lifetime dependency |
| 1. Hardware proof | Isolated HIP peer-visibility/collective and launch/graph probe, widths 2, 3, 4; RCCL F32 sum and byte-transfer comparison; initial wave32/gfx1151 compile probe | Ordered-sum oracle, visibility/liveness tests, latency distributions and rank skew; explicit verdict on fine-grain prerequisite and RCCL path eligibility |
| 2. Backend foundation | Optional build/runtime loading, buffers, tickets, copies, allocation accounting and errors; basic ops and dense F32 reference execution | Native failure/lifetime tests; loader modes; tiny HF model; CPU/Vulkan unchanged |
| 3. First useful model | Qwen3 Q8_0 prompt/decode, paged attention, caches, requested dtypes and follow-up/server path | Independent HF short/depth checks, batch/row identity, one-card matched perf screen; qualify only implemented paths |
| 4. Tensor execution | F32 fallback, qualified custom collective and RCCL paths if adopted; two/three/four members, stages of groups; bounded decode graph replay | Group correctness, graph/eager identity, skew/failure tests and first dense mx matrix with all gaps shown |
| 5. Complete operations | Remaining active quant registry, grouped/routed calls, hybrid recurrence/state, embedding/head, embedded draft and rollback | Per-family numerical witnesses and HF budgets; mixed batches, draft on/off, split/restore/server tests |
| 6. Performance closure | Tune measured hot kernels, packing, attention, collectives and graph overhead at default clocks | Full same-machine mx matrix, existing-path tradeoff assessment, archived profiles and no hidden missing cells |
| 7. Future hardware | Strix Halo bring-up when available; later CDNA/RDNA devices in independent branches | Per-device correctness/perf matrix, UMA pressure tests and measured profile; otherwise explicitly untested |

Steps 2-5 may interleave by complete feature: after the buffer/execution contract
is fixed, one owner can build runtime/collectives, a second kernels/attention and
a third tests/profiles. Shared interface/dispatch edits have one integrator.
Independent correctness runs can share host/GPU resources when coordinated;
performance arms must avoid competing work on the assigned resources and monitor
unrelated activity. No developer reruns another developer's retained cells merely
to occupy waiting time. Remove answered experiment knobs and unreachable kernels.

For each kernel candidate: focused correctness and dispatch witness, short
matched performance screen, bottleneck diagnosis if it loses, then expensive
HF/depth qualification for candidates worth integrating. This short screen does
not replace the complete correctness or final performance matrix.

## Final gate matrix and reporting

Freeze the actual model hashes and reference commands before runs. Cover dense
Qwen3 (0.6B, 8B, 32B), supported hybrid dense and MoE models, Q8_0/K-quants/MXFP4,
and the other active storage families through their fixtures. Use the existing
precision policy; no new tolerances in the backend. The model/quant list is
updated explicitly at the start if main has gained support since this plan.

| Dimension | Required cells |
| --- | --- |
| Hardware/topology | One MI50; tensor widths 2, 3, 4; staged groups and 8 cards under the existing topology rules; later Strix Halo separately |
| Precision | Default F16 and explicit F32/BF16/int8 correctness/fallback; matched-int8 speed arm against mx plus default beside it; record cache, accumulation and collective precision |
| Single user | Prefill 512/2048/4096/16384 and decode 128 at declared history depths; fresh, follow-up and restored histories |
| Serving | 1/4/16/32/64 users, short/long/mixed/skewed arrivals, cancellations and drafting off/on where supported |
| Metrics | Prefill and decode tok/s, output tok/s, TTFT, inter-token p50/p99, CPU time/token, memory peak and load throughput |
| Correctness | Native ops/range/lifetime; independent file-exact HF logits/NLL/rankings/depth; cache/state operations; same-backend batch invariance; intended group-order tolerance rather than invented cross-width bit equality |

Do not claim that tensor groups must equal a single device bit for bit: splitting
a reduction can change its arithmetic, especially routed choices. Preserve the
existing tensor gate and HF checks, and require identical member results within
a collective and identical results for equivalent batch/row classes. Existing
int8 quality exceptions remain named costs, not newly passing F16 cells.

For every rate report: model/hash, topology, precision/path, llmx before/after,
mx target, ratios, repeated-run spread and activity flags. For latency, lower is
better. Show per-cell passes/open cells, never only an average. The user's goal
is to beat mx at each required topology/workload; an uncertain difference is
inconclusive, and a first-support merge allowed by current policy is not that
goal's completion. Material tradeoffs are presented under AGENTS rather than
discarding a large gain for one noisy small loss.

Use matched release builds, hashes and versions; record code-object targets,
runtime, driver, topology and clocks. Retain every planned interleaved block and
timestamped CPU/process/disk/GPU monitoring. Profile separately from timing;
summarize launch gaps, barrier wait, memory traffic, occupancy and spills as
text/CSV/JSON. Existing Vulkan timings are context, not a ROCm baseline.

Full release also needs the applicable complete suites, docs/dead-code and
architecture checks, hosted CI at the exact head, Linux HIP build coverage and
CPU/Vulkan nonregression. Hosted no-GPU jobs cannot establish device gates.
Shared changes retain the Windows Radeon VII checks; ROCm itself is not claimed
to execute on that Windows machine. Builds for gfx1151 without hardware are
labeled compile-only.

## Open decisions and checkpoint reporting

Step 1 resolves the fine-grain memory prerequisite and actual custom-collective
gain, including RCCL path eligibility. The user approved the optional dependency
and the early probe on 2026-10-09. XDEV owns the probe; the full backend's
implementation order remains unchanged. Freeze workloads and thresholds in the
collaboration log before running. Use the existing community ROCm image and
cards whose PCI root complex is separate from production's cards, after checking
their PCI paths. Reserve a named resource window with START and END records,
bounded supervision and no device resets; install nothing on the host.
Step 2 resolves portable code-object packaging and the minimum supported
HIP API/toolchain. Step 4 resolves graph-boundary adequacy with a working trace,
not speculative interface expansion. Strix memory behavior and tuning wait for
hardware. None of these open items is implemented by this document.

Pin the community gfx906 image by digest, its runtime/compiler versions and
recoverable build sources; do not depend on a moving tag as the support policy.
First-support milestones follow AGENTS: open speed cells are recorded with
immediate recovery work, never called a passed phase-final gate or used to hide
a regression on existing paths. The completion scope remains all current
operations/types; a Q8_0-only milestone refuses missing capabilities explicitly.
Early loader, fit and lifetime checks stay in the foundation milestone. They
cannot be postponed until after model execution and tensor groups are shipped.

At each checkpoint report implemented, validated, merged, untested and open
performance cells separately, with the next bounded task and executable path.
Re-estimate effort after steps 1, 3 and 4. Earlier calendar estimates are rough
planning ranges; source research cannot promise when every mx cell will pass.
