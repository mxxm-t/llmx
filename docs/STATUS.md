# llmx - Development Status

## Matched mx attention exposes remaining compute gaps (2026-10-10, docs only, lands by fast-forward)

- **Done:** the [attention follow-up](benchmarks/rocm-attention-20261010/README.md)
  measures unchanged Vulkan, vector HIP and shipped mx HIP on identical logical
  fixtures in one environment. HIP/mx latency is 0.98x at 512 fresh rows, 1.51x
  at 2048, 1.83x at 4096 and 1.94x at 512 rows after 2048 history, in both
  orders. Full numbers and retained errors accompany the result.
- **Finding:** traces prove mx's GQA-sharing tile, combine, long-prompt mask
  helper and graph replay. Its narrower arithmetic even makes a singleton
  causal row inexact; the failed harness assumption and independent correction
  are retained. No llmx numerical bound changes, and no speed gap is closed
  merely by measuring the reference.
- **Checks:** 108 smoke, 15 trace/control and 24 timing processes pass their
  stated fixture contracts. All samples remain; timing has no activity flags,
  complete coverage and restored starting VRAM. Ordinary finite fixtures are
  not full-model, HF or range qualification. The immutable archive is mirrored
  and verified on rig and workstation.
- **Docs review:** all tracked Markdown inventoried and local links checked;
  changed ROCM, STATUS and benchmark claims reconciled with raw measurements,
  trace dispatch and pinned reference source. Other pages retain the preceding
  review: runtime, CLI, build, precision, tests and ownership are unchanged.
  Exact-tree docs/dead-code pass; this docs-only tier needs no hosted wait.
- **Left:** isolate GQA reuse and tiling with current F32 arithmetic, then the
  combined compute/collective admission and full backend release gates. Core
  dtype remains complete; CPU emulation speed stays nonblocking. No production
  ROCm implementation or new Windows executable is delivered here.

## ROCm attention measured and exact KV staging improved (2026-10-10, docs only, lands by fast-forward)

- **Done:** the private full-attention probe, isolated QK/PV layouts, actual
  dispatch traces and full-kernel phase clocks are measured. Load plus barrier
  takes 37-42 percent of recorded workgroup cycles, with 5-8 percent profiling
  overhead. Wider exact KV loads then cut full HIP latency by 18-25 percent.
  The candidate beats Vulkan by about 12 percent at 512 fresh rows, stays about
  2 percent behind with history, and remains 20-23 percent behind at 2048/4096.
- **Validation:** 114 candidate correctness processes, 19,906,560 independent
  oracle checks and 152 bit-identical whole-output comparisons pass. All 32
  matched timing processes pass, with both orders and every sample retained.
  The earlier phase matrix's four flagged calls and two flagged measured chains
  remain; inaccessible process activity is unknown. gfx906 and gfx1151 compile
  without spills, with gfx1151 still untested on hardware. Sources, plans,
  negative results, monitors and traces are archived in the
  [attention record](benchmarks/rocm-attention-20261010/README.md).
- **Docs review:** all tracked Markdown inventoried and local links checked;
  ROCM and this record reconciled with the measured sources, plans and results.
  README, architecture, roadmap, precision, build, usage, CI and source-owner
  pages retain their current CPU/Vulkan scope; no production ROCm support or
  build option is added. Historical measurements retain their stated scope.
  Exact-tree docs and dead-code checks pass; runtime and tests are unchanged.
- **Left:** matched mx attention, remaining long-prompt cost and combined
  compute/collective admission, then full model/HF/range/lifetime/split gates
  before production integration. Core dtype remains complete; CPU emulation
  correctness is required, but its speed is nonblocking.

## Dtype test guidance refreshed (2026-10-10, docs only, lands by fast-forward)

- **Done:** BUILD lists `int8` for the test runner and split tool and describes
  the split tool's optional tensor width after dtype, matching their parsers.
  PRECISION records the user's performance priority: GPU and native CPU
  arithmetic; CPU emulation remains correctness-covered and its speed is
  nonblocking. Core dtype stays complete; no runtime policy or bound changes.
- **Review/checks:** all Markdown inventoried and local links checked; the
  changed claims follow `tests/run_tests.py`, `tools/split_check.cpp`, current
  precision reporting and the user's clarification. Historical measurements
  retain their scope. Docs/dead-code run on the landing tree; no hosted wait
  applies to this documentation-only tier.
- **Left:** no work in this correction. Separate GPU performance, ROCm and
  future native BF16 support retain their existing scope and gates.

## ROCm Q8 phases and row layout measured (2026-10-10, docs only, lands by fast-forward)

- **Goal:** locate the remaining compute cost before the full backend's
  admission decision; complete the bounded Q8 diagnostic work.
- **Done:** [phase and row-layout follow-up](benchmarks/rocm-q8-compute-20261010/README.md)
  records 56 phase processes and 32 matched row-layout processes, with all
  numerical/path/guard checks passing. Preparation is about 1.8 us; product
  work dominates. The six-wave hint spills and loses, and one row per wave
  loses despite fewer registers. Those controls are retained and rejected.
- **Finding:** uniform two-row ownership lowers F16 latency by 6.2-17.7 percent
  but increases int8 latency by 4.9-14.2 percent. Keep the prior int8 candidate
  separately. The F16 gap to Vulkan remains about 8-13 percent; no mx kernel
  or model gate is passed. The phase matrix's one flagged CPU call and four
  flagged measured chains stay; row-layout timing has no declared flags.
- **Left:** full prompt attention next, then the combined admission decision;
  the remaining compute cost and full backend/HF/lifetime/mx gates stay open.
  gfx1151 remains compile-only. No runtime source or backend option is added.
- **Review/checks:** all Markdown inventoried, local links and the tree's
  documentation/dead-code components checked; changed claims reconciled with
  precision, architecture, build, roadmap and raw evidence. Historical results
  retain their scope. No hosted wait applies to this documentation-only tier.

## ROCm Q8 compute measured (2026-10-10, docs only, lands by fast-forward)

- **Complete:** [Q8 compute and profile record](benchmarks/rocm-q8-compute-20261010/README.md)
  retains 24 original and 40 matched candidate processes, 1280 measured chains
  and 192 warmups. All bounded numerical/path/guard checks pass; no declared
  contention flags or missing GPU counters. Sources, traces and all samples stay.
- **Finding:** consecutive raw-weight loads reduce HIP graph latency by 28-41
  percent, but retain 9-31 percent extra latency versus Vulkan. No standalone
  mx kernel is measured. Profiling doubles elapsed time, so its marked kernel
  breakdown is diagnostic and the unprofiled controls carry speed results.
- **Scope:** ordinary finite Q8 fixtures, canonical blocks, F16/int8 separately;
  gfx906 execution, gfx1151 compilation only. No production backend or option.
- **Left:** resolve the remaining compute cost and run full prompt attention
  before admission; full model/HF/lifetime and mx gates remain open.
- **Review/checks:** all Markdown inventoried and local links checked; current
  ROCm, precision, architecture, build and roadmap claims reconciled with this
  bounded scope. Historical results retain their scope. Docs/dead-code run on
  the landing tree; no hosted wait is required for this documentation tier.

## ROCm model budget refreshed (2026-10-10, docs only, lands by fast-forward)

- **Complete:** [matched model matrix](benchmarks/rocm-model-budget-20261010/README.md)
  runs all 20 processes, 120 measured phases and 40 warmups on Qwen3-32B Q8_0,
  widths two and four, default F16 and explicit int8 separately, against shipped
  mx in the same environment. All samples and setup failures are retained.
- **Diagnostic:** skipping only one-row sums preserves prompt hashes and
  deliberately changes decode output. The report gives the refreshed time
  available per sum beside the previous HIP graph screen. This is a planning
  estimate, not a correct model result or passed ROCm/HF gate.
- **Scope:** every llmx phase records executed precision, both runtimes return
  completed logits inside timing, model hashes agree before/after and per-phase
  activity flags remain visible. The report preserves the mx precision and
  communication-policy differences, warnings and the depth-zero workload limit.
- **Next:** Q8 decode compute and full prompt attention still need their bounded
  probes before admission. No production backend or build option is added.
- **Review and checks:** all Markdown is inventoried and local links checked;
  changed owners are reconciled with the retained samples, protocol and source.
  Prior status records are preserved. Documentation/dead-code run on the tree
  that lands; no hosted wait applies to this docs-only tier.

## Dtype and split guidance correction (2026-10-10, docs only, lands by fast-forward)

- **Done:** PRECISION now identifies the activation dtype implementation as
  released, with qualification and separate speed gaps in this status record.
  AGENTS describes the implemented tensor groups selected by `--tensor-width`
  and distinguishes planned multi-node execution. The old sentences described
  completed work as pending; no runtime behavior or gate is changed.
- **Evidence:** the activation release `8af97e88`, Vulkan int8 `c8b2ac8d` and
  its repair `f80709f20` are ancestors of the starting main `fb8240433`.
  The split wording follows the current execution options and MULTI-DEVICE.
- **Review:** all project Markdown is inventoried and its local links checked;
  these two live claims are corrected against the current owners. Historical
  qualification records remain unchanged. This is a focused semantic review,
  not a new verification of every historical measurement.
- **Checks:** documentation and dead-code on the tree that lands. No runtime,
  test or build changes; no hosted wait applies to this documentation-only tier.

## ROCm direct reference and transport controls (2026-10-09, docs only, lands by fast-forward)

- **Complete:** [matched control matrix](benchmarks/rocm-controls-20261009/README.md)
  adds fresh shipped-mx eager/graph calls and conservative HIP-copy/RCCL-transfer
  arms beside the retained custom HIP and current Vulkan controls. All 168
  planned processes pass, 3,360 measured chains and 504 warmups, after 24 smoke
  processes. All samples, failures during setup, path witnesses and teardown
  checks are retained and verified locally and on the rig.
- **Measured:** at width four and 5,120 F32 values, custom graph is 24.50/23.77
  us per sum against mx graph 23.89/24.08, including the residual in both.
  With four dependent increments they are 29.07/29.32 and 29.75/28.91.
  The larger-vector width-two/three custom losses, every eager/conservative
  row and both block medians are in the linked matrix. These are collective
  microbenchmarks, not model throughput or passed phase-final mx gates.
- **Activity:** 96 calls overlap the declared unrelated-CPU flag, all 56 at
  width four included; observed package/compiler work peaks at 15.02 cores.
  No GPU counters or leading/trailing coverage are missing. All results stay;
  there is no small-win claim or replacement of affected samples. Every arm
  uses the same fine-grain vendor prerequisite and hardware-queue setting;
  earlier unset-environment figures are separate.
- **Next:** refresh the model budget before a ROCm admission decision. Q8
  decode, full prompt attention and the full-backend gates remain open.
  No production backend/build option is added. Probe resources are released.
- **Checks:** documentation and dead-code pass on this tree; the linked record gives
  the bounded experimental checks and their limits. Historical main records
  are preserved. All 103 Markdown pages and 512 local targets are inventoried;
  the three changed pages are checked against the retained evidence. This
  checkpoint changes only documentation and lands as one commit by fast-forward.

## ROCm research and delivery plan (2026-10-09, docs only, lands by fast-forward)

- **Prepared:** [ROCM](ROCM.md) defines the existing owners, optional HIP runtime,
  kernels, collectives, graph lifetime, numerical policy and checkpoint gates.
  [Pinned research](research/rocm-20261009.md) compares mx custom all-reduce and
  compute dispatch with vLLM, SGLang/AITER and Megatron Core. Vulkan's capability
  and profile structure is reused; its measured RADV choices are not HIP defaults.
- **RCCL correction:** evaluate custom AR plus optional RCCL for large F32 sums,
  byte-preserving stage transfers and fallback. The earlier blanket exclusion
  was unjustified: mx uses both. Adoption requires the pinned MI50 runtime,
  numerical, lifetime and performance checks; the user authorized its separate
  off-by-default build option. No dependency has been added.
- **Peer proposal reconciled:** the 2026-10-08 research is reviewed in the
  research record and folded into the same plan. Historical sum budgets are
  calculated from one session and labeled estimates; eager/graph launch, decode,
  sums and full attention have distinct probes. HRX/direct packets are separate
  proposals, and loader/fit/lifetime checks remain early.
- **Future hardware:** gfx1151/Strix Halo is part of the design through wave32
  portability and compile coverage of kernels already being written. Shared-memory
  accounting and profiles wait for hardware. Compilation is not a support claim.
  MI50/Linux is the first execution target.
- **Deployment clarification:** vendor runtime and driver environment variables
  may be documented prerequisites, explicitly set in the backend image and
  reported with the selected fast path or fallback. llmx's runtime knobs stay
  flags; temporary development variables do not ship as hidden controls.
  ROCm is an off-by-default build option beside Vulkan, with an independently
  optional RCCL build option under it; both config forms stay in sync.
- **Probe checkpoint:** a standalone 500-dependent-add launch smoke passes
  under HIP eager submission, HIP graph replay and the existing Vulkan backend
  on one MI50: each checks 64 outputs equal to 500 in all 25 chains, 4,800
  checked values in total. It uses PCI 0000:89:00.0 on root pci0000:80, separate
  from production's pci0000:c0, with bounded processes and no peer barriers.
  The community image is pinned to SHA256
  `1e1a116b443f2b474e7fe14933552013ef68003364d7ec194d6101efa92faa87`,
  HIP 7.2.53211 and clang 22. Both probe sources build with warnings as errors.
  Concurrent correctness tests make its recorded timings diagnostic only;
  this proves neither a speed gain nor peer visibility. Source, hashes and
  outputs are in `/zpool1/llmx-xdev-validation/rocm-probe-20261009`.
  The same standalone source also compiles for gfx1151 with warnings as errors;
  that is compile coverage of this launch probe only, with no Strix Halo run
  or backend-support claim.
- **Peer probe build:** the standalone peer-memory check compiles independently
  for gfx906 and gfx1151 with `-O3 -std=c++17 -Wall -Wextra -Werror` in the
  same pinned image, without access to any GPU. It tests public ordinary,
  fine-grained and uncached allocations, rotating delayed submissions, guarded
  peer writes and fixed-order F32 sums after host synchronization. Source and
  binary hashes and both compile logs are in that probe directory's
  `build-peer-1`. Compilation alone qualifies no peer-memory run, device-side
  visibility protocol, collective or RCCL; the hardware checks below are separate.
- **RCCL inventory:** that image has installed `rccl` and `rccl-dev` packages
  `2.27.7.70201-81~24.04`. Their header and library live under `/usr/local`,
  outside the `/opt/rocm` prefix. The header identifies 2.27.7, the loader
  resolves the library and its dependencies, and `ncclGetVersion` returns
  success with 22707 without any GPU exposed to the container. Header/library
  hashes and all inventory attempts are retained in `rccl-inventory*.log` in
  the probe directory. This establishes availability for the planned probe,
  not MI50 communicator, graph or numerical qualification.
  `roc-obj-ls` could not inspect the library's embedded targets because the
  image lacks Perl's `File::Which`; `rccl-targets.log` retains that tooling
  failure, from which no target-support conclusion is drawn.
- **Peer hardware checkpoint:** groups of two, three and four MI50s pass all three
  public allocation modes, with the frozen source/binary and no environment
  override. Each cell runs 1, 127 and 5,120 values through 16 epochs apiece.

  | Members | Rig GPUs (PCI suffixes) | Ordinary | Fine-grained | Uncached |
  | --- | --- | --- | --- | --- |
  | 2 | 2/3 (83:00.0, 86:00.0) | 48/48 epochs | 48/48 epochs | 48/48 epochs |
  | 3 | 2/3/5 (83:00.0, 86:00.0, 8c:00.0) | 48/48 epochs | 48/48 epochs | 48/48 epochs |
  | 4 | 2/3/4/5 (83:00.0, 86:00.0, 89:00.0, 8c:00.0) | 48/48 epochs | 48/48 epochs | 48/48 epochs |

  All directed peer links are available. Every peer payload, guard and local
  fixed-order F32 sum matches exact bits, including cancellation-sensitive
  three- and four-member sums and rotating delayed submissions. All 432 epochs pass.
  `HSA_FORCE_FINE_GRAIN_PCIE` is unset; host stream synchronization separates
  writes and sums. This does not establish device-side signal visibility,
  barriers, graph replay, RCCL collectives or performance. All processes drain
  and each card's VRAM returns to its recorded pre-run value. The nine raw
  `peer-2-*-1`, `peer-3-*-1` and `peer-4-*-1` directories and `peer-summary.json`
  retain the checks under the probe directory. Width four ran after the peer
  developer released its fourth card; production's PCI root remained excluded.
- **RCCL numerical screen:** the standalone probe compiles for gfx906 and
  gfx1151 with warnings as errors; linking it says nothing about the library's
  gfx1151 support. The MI50 runs use the pinned RCCL 2.27.7, F32 data, default
  algorithm/protocol selection and `NCCL_DEBUG=INFO` for the retained log.
  Each cell completes all 48 epochs at 1, 127 and 5,120 values, using the same
  changing inputs, guards, rank delay and independent member-order F32 oracle.

  | Members | Send/receive plus ordered sum | Native all-reduce exact epochs | Native sum mismatched values |
  | --- | --- | --- | --- |
  | 2 | 48/48 exact | 48/48 | 0 |
  | 3 | 48/48 exact | 0/48 | 42,096 |
  | 4 | 48/48 exact | 0/48 | 63,040 |

  The native three- and four-member cells return 3, the declared numerical
  failure, after completing every planned epoch. For the one-value positive
  cancellation case, member order gives 0 and 3 respectively; RCCL gives 1
  and 4. Different addition order explains why F32 alone cannot promise this
  contract; this is not a claim that every RCCL result is less accurate.
  Guard and transfer checks pass. Every size's communicator set initializes
  and destroys normally; all six containers drain and each card's VRAM returns
  to its measured starting value. The raw `rccl-*-*-1` records and independent
  `rccl-summary.json` retain successes and failures. No tolerance or wire
  precision was changed. Native sums at three and four members in this setup
  are ineligible for the current ordered-sum contract; the transfer path stays
  a candidate. Two-member success is scoped to these inputs. Graph replay,
  failure recovery, arbitrary bit-pattern transfers, size crossovers and
  matched performance remain unqualified, as does backend adoption.
- **Device-signal checkpoint:** the independently written one-block probe uses
  uncached system-release/acquire epoch flags, an explicit system fence from
  every peer writer, original local contributions and a consumed barrier before
  staging reuse. Shapes 1, 127, 5,120 and 5,121 run 32 changing epochs inside
  each kernel and four launches per shape over the same buffers, with rotating
  2 ms rank launch delays. No host wait separates the in-kernel phases.

  | Members | Uncached payload | Fine-grained payload | Ordinary payload |
  | --- | --- | --- | --- |
  | 2 | 512/512 exact epochs | Stale payload, exit 1 | Stale payload, exit 1 |
  | 3 | 512/512 exact epochs | Stale payload, exit 1 | Stale payload, exit 1 |
  | 4 | 512/512 exact epochs | Stale payload, exit 1 | Stale payload, exit 1 |

  Each passing epoch checks every consumed contribution independently, every
  ordered F32 result, unchanged inputs, guards and the untouched self slot.
  All 1,536 uncached epochs pass with `HSA_FORCE_FINE_GRAIN_PCIE` unset.
  Every failed cell completes its first 32 signal handshakes but reads the old
  `0xa5a5a5a5` canary instead of 1.0 at the first checked remote contribution;
  it stops at that mismatch, so none of its epochs is counted as validated.
  The six failures remain in the evidence. They show why the earlier
  host-synchronized pass does not establish in-kernel payload visibility.
  Ordinary allocation is only a diagnostic control, even had it passed.
  Missing-member controls at widths two and four omit the final member's
  launch: every launched member exits at its bounded ready-epoch-1 wait,
  completing no epoch. The controls also check untouched outputs and guards.
  Polling is bounded by both 2^24 loads and 500 million shader-clock ticks;
  these are not wall-clock guarantees. The process has a 120-second timeout
  and a ten-second kill bound. This is a specific timeout check, not general
  failed-device recovery. Every process drains, its container disappears and
  all cards return to their exact pre-run VRAM use, without resets.
  Both targets compile with warnings as errors. Source SHA256 is
  `2ece4f597acfa15edc0922eb8af191aa33a1bb63e08b3b94cd9d089cfd68a8fc`,
  gfx906 binary `ed6bd8cc5c6a1e9a1126e612871cd67dc8a33a584ab96a34bfb73ed91fe4680f`.
  Emitted assembly records 104 SGPRs, 39 VGPRs and 72 scratch bytes on gfx906;
  gfx1151 records 62, 33 and 72 respectively, with no hardware claim.
  Despite the source non-temporal hint, gfx906 payload loads are ordinary
  global loads. No cause beyond the measured allocation-mode difference is
  established, and this diagnostic kernel is not performance-ready.
  The first assembly command failed because the hipcc wrapper added an unused
  link flag under warnings-as-errors; direct clang generated both assemblies
  from the unchanged source. Both attempts are retained. Raw `signal-*-1`
  records, `build-signal-2`, the frozen `signal-protocol.md` and the independently
  checked `signal-summary.json` are under the same probe evidence directory.
  Uncached staging is the next candidate; graph replay, multiple blocks,
  message-size crossovers, model execution and matched latency remain open.
- **Multi-block graph checkpoint:** a separate standalone probe uses uncached
  payloads and signals, per-block ready/consumed epochs and an ordered chain
  of 32 separate collective kernels. The same kernel runs through eager
  submission and captured graph replay. A stable device descriptor changes
  the base sequence between four launches while every payload changes.
  Each block owns the same indices on all members, including empty blocks,
  and waits only for that block on peers. Grids 1, 2, 4 and 16 are below the
  queried theoretical residency limit; shapes are 1, 127, 5,120, 5,121 and
  65,537 values. A rotating member launch is delayed 2 ms per batch.

  | Members | Eager exact epochs | Graph exact epochs |
  | --- | --- | --- |
  | 2 | 2,560/2,560 | 2,560/2,560 |
  | 3 | 2,560/2,560 | 2,560/2,560 |
  | 4 | 2,560/2,560 | 2,560/2,560 |

  All 15,360 normal epochs match the same independent ordered F32 oracle,
  including every saved contribution, result, guard, unchanged input, untouched
  self slot and final block sequence. Thus eager and replay agree exactly on
  these inputs. Each of the 192 captured member graphs, including the negative
  controls, has 32 kernel nodes; capture and instantiation leave all initialized
  buffers and counters unchanged. Every successful replay advances each block
  by exactly 32 epochs. Host waits occur around batches, not between kernels.
  Graph controls at widths two and four omit the final member: all 64 active
  blocks exit at their bounded ready-epoch-1 wait and no result is written.
  Two other controls first replay successfully, then repeat base sequence zero:
  all 96 blocks refuse the stale epoch before writing payloads or flags, retain
  the earlier staging/sequence and leave new outputs untouched. Their two
  successful preludes add 64 exact epochs separately from the normal matrix.
  A failed block skips later kernels. Poll and process bounds are unchanged
  from the signal probe. Every group drains all members before graph or buffer
  destruction; all ten processes exit zero, containers disappear and cards
  return to the exact pre-run VRAM use. No reset or production device is used.
  Source SHA256 is
  `9acf65c2ec7c911aaaf61b12c70bf3dafc80ff0ef7f11f9ecc626bc2bda49be1`,
  gfx906 binary `09eeb6bfbab2872a3937a7568a2f89ff7f427fe3b52dfc540c4191edd12d5133`.
  Both targets compile with warnings as errors. Assembly records gfx906 at
  96 SGPRs, 15 VGPRs and 72 scratch bytes; gfx1151 at 70, 21 and 72, with no
  hardware execution claim. `graph-protocol.md`, `build-graph-1`, ten raw
  `graph-*-1` directories and the locally and remotely verified
  `graph-summary.json` retain the evidence. The vendor fine-grain override
  remains unset. This qualifies this bounded probe, not model graph caching,
  arbitrary shape/allocation reuse, general recovery or performance. The
  diagnostic copies of every contribution are not a production cost model.
- **Lean residual checkpoint:** a separate candidate removes those diagnostic
  copies, takes the helper's target descriptor by const reference and updates
  one residual in place with the member-ordered F32 sum. It matches
  `Collective::sum_into`'s residual addition, while inputs still come from a
  prepared 32-entry array rather than model kernels. Final residuals match an
  independent volatile F32 oracle in every chain of the same frozen matrix.

  | Members | Eager checked chains | Graph checked chains |
  | --- | --- | --- |
  | 2 | 80/80 | 80/80 |
  | 3 | 80/80 | 80/80 |
  | 4 | 80/80 | 80/80 |

  These are 480 normal chains of 32 kernels, 15,360 completed kernels; only
  each chain's final residual is checked, not saved intermediate results.
  Inputs, final staging, guards, signals and statuses also pass. All 192
  captures leave initialized storage unchanged, and the four width-two/four
  missing-member and stale-base controls pass. Their two successful preludes
  are separate. All ten processes drain and restore the exact starting VRAM.
  Both targets compile with warnings as errors. gfx906 now uses 49 SGPRs,
  12 VGPRs and zero scratch bytes; gfx1151 uses 40, 10 and zero. Neither
  assembly has private buffer loads/stores. This combined simplification
  removes the diagnostic scratch cost; no isolated attribution or measured
  speedup is claimed. The original graph probe remains unchanged.
  Source SHA256 is
  `10f23d3e9cc15622d57823302e9dd1e5a2c64c88572503dd1fd2d4cb6b1e7430`,
  gfx906 binary `3e0541ae1f4c691c04bc637a716008412e8753aac5d16df0b0ec4cb7f0fa7913`.
  `lean-protocol.md`, `build-lean-1`, ten raw `lean-*-1` directories and
  `lean-summary.json` retain the evidence in the same probe directory.
  No gfx1151 execution, model correctness or performance qualification follows.
- **First matched timing checkpoint:** the current-main Vulkan collective and
  independent HIP eager/graph harnesses build with ROCm clang 22 and run in
  one image, SHA256
  `d5591273e00b5735bd3419009ffd7575821dcf957897010e03ba1d1c6a0ae14f`.
  It has HIP 7.2.1, Mesa 25.2.8 and shaderc 2025.2. This is a fresh control,
  not a subtraction from the earlier Mesa 25.0.7 image. The initial distro
  shader compiler lacked the integer-dot extension; its failed build and a
  subsequent local-image lookup failure remain retained. The successful
  build uses the project's existing shader compiler in the common image.
  The Vulkan source is clean main `f80709f20`; the rebuilt llmx reports
  `llmx 0.1.0+gf80709f204a8 numerics 7cbc01b7c270d329`.
  The frozen matrix has two forward/reverse arm blocks, 128 sums per chain,
  three warmup and 20 measured chains per process, sizes 5,120 and 65,537,
  with zero or four dependent increments after each sum. HIP uses 16 blocks
  and one persistent host thread per rank; Vulkan uses its existing owner.
  All 72 processes, 1,440 measured chains and 216 warmups pass final output
  and guard checks. HIP block statuses/sequences pass too. Intermediates are
  not saved; constant per-rank inputs exercise repeated submissions, not
  changing model activations. Both targets compile; only gfx906 executes.

  Host wall time per sum or sum-plus-increments unit, pooled median in us,
  lower better; the material block spread below is part of these results:

  | Members | F32 values | Increments | Vulkan | HIP eager | HIP graph | mx |
  | --- | --- | --- | ---: | ---: | ---: | --- |
  | 2 | 5120 | 0 | 92.65 | 11.87 | 10.65 | Not remeasured |
  | 2 | 5120 | 4 | 102.68 | 20.26 | 16.08 | Not remeasured |
  | 2 | 65537 | 0 | 95.78 | 54.00 | 52.72 | Not remeasured |
  | 2 | 65537 | 4 | 100.34 | 63.42 | 59.95 | Not remeasured |
  | 3 | 5120 | 0 | 146.35 | 17.83 | 16.64 | Not remeasured |
  | 3 | 5120 | 4 | 146.19 | 26.47 | 22.04 | Not remeasured |
  | 3 | 65537 | 0 | 150.22 | 107.99 | 106.85 | Not remeasured |
  | 3 | 65537 | 4 | 160.02 | 117.63 | 114.12 | Not remeasured |
  | 4 | 5120 | 0 | 222.91 | 24.75 | 24.19 | Not remeasured |
  | 4 | 5120 | 4 | 230.36 | 40.99 | 28.96 | Not remeasured |
  | 4 | 65537 | 0 | 229.44 | 155.69 | 154.55 | Not remeasured |
  | 4 | 65537 | 4 | 241.43 | 165.37 | 161.79 | Not remeasured |

  Width four's small graph sum has process medians 23.87 and 48.70 us;
  its eager mixed chain has 34.22 and 58.67 us. Keep both, rather than read
  the pooled 24.19 as a stable sum budget. The graph mixed medians are
  29.02/28.88 us, and Vulkan's small-sum medians 222.67/223.79 us.
  The slow graph block's median rank-start skew is 3,040.51 us against
  7.78 us in the first block, while median chain wall time is 6,233.91
  against 3,055.40 us. This locates a host-launch delay in the measurements;
  it does not yet establish why a worker starts late. Compare the host wait
  policy next, preserving this baseline and the same workload.
  Mixed increments are placeholders, a scalar HIP increment versus Vulkan's
  ones buffer, not equivalent model compute. Device events, enqueue time,
  per-process medians/p95/ranges and every sample remain separate in the data.
  The one-second monitor covers all processes, with no GPU counter missing
  and no leading/trailing coverage gap. One of 72 processes is flagged for
  unrelated CPU: width three, 65,537 values, four increments, reverse-block
  Vulkan, peaking at 3.95 observed cores. All results remain included.
  No recorded flag explains the width-four spread; sampling cannot attribute
  individual millisecond chains, and transient/inaccessible processes remain
  coverage limits. Maximum observed monitor sample overhead is 68.35 ms.
  All containers are removed and exact starting VRAM restored. Evidence in
  the same probe directory: `latency-protocol.md`, `latency-plan.json`,
  `build-latency-2`, `latency-timing-1`, the three `latency-activity-1-w*`
  directories and `latency-summary.json`, verified locally and on the rig.
  Fresh mx controls, conservative HIP/RCCL alternatives and a refreshed model
  budget remain required before an advance verdict or a full-model speed claim.
- **Host-wait checkpoint:** one executable selects per-rank waiting or group
  waiting, where every rank finishes enqueueing before the controlling thread
  waits for any member. Both policies use the same kernels. Whole emitted GPU
  assembly matches the previous harness on gfx906 and gfx1151 after removing
  only the source filename and compiler-generated unit ID; the first verifier's
  rejection of that ID is retained, with the corrected check and unchanged
  binaries. No instruction or resource difference is normalized away.
  Four MI50s, 5,120 F32 values, the same 128-sum chains and forward/reverse
  arm blocks give these process medians in us per sum or mixed unit:

  | Increments | Arm | Block 1 | Block 2 | mx |
  | --- | --- | ---: | ---: | --- |
  | 0 | Vulkan | 215.47 | 217.47 | Not remeasured |
  | 0 | HIP eager, rank wait | 24.78 | 24.69 | Not remeasured |
  | 0 | HIP eager, group wait | 24.93 | 24.80 | Not remeasured |
  | 0 | HIP graph, rank wait | 23.77 | 30.61 | Not remeasured |
  | 0 | HIP graph, group wait | 23.52 | 23.57 | Not remeasured |
  | 4 | Vulkan | 228.56 | 226.06 | Not remeasured |
  | 4 | HIP eager, rank wait | 57.98 | 36.40 | Not remeasured |
  | 4 | HIP eager, group wait | 33.36 | 48.58 | Not remeasured |
  | 4 | HIP graph, rank wait | 28.90 | 28.97 | Not remeasured |
  | 4 | HIP graph, group wait | 29.10 | 29.15 | Not remeasured |

  All 20 processes, 400 measured chains and 60 warmups pass output, guard,
  status and sequence checks. The small graph rank-wait control has median
  launch skew 5.50/819.78 us, against group waiting's 32.66/27.86 us. The
  earlier exact 3 ms graph delay did not recur; eager mixed rank waiting has
  3,048.86 us skew in its first block, and group eager still varies. Thus the
  consistent graph group-wait result supports that submission ownership here,
  not a claim that every scheduling delay is fixed. All samples are kept.
  No process triggers the frozen activity flags; maximum observed unrelated
  CPU is 0.55 core, with the same monitoring limits as before. All containers
  are removed and starting VRAM restored. `latency-wait-protocol.md`,
  `latency-wait-plan.json`, `build-wait-1`, `latency-wait-timing-1`,
  `latency-wait-activity-1-w4` and the locally/remotely verified
  `latency-wait-summary.json` retain the evidence. No production code changes.
  Use submission before waiting for the next controls; fresh mx, conservative
  HIP/RCCL comparisons and the model budget remain open before admission.
- **State:** this branch adds documentation only; no backend is implemented.
  The existing
  full-backend implementation order and closed kernel/driver-patch route remain.
  The user authorized an earlier bounded probe, owned by XDEV, on a separate
  PCI root complex from production, with fixed workloads and no device resets.
  The next hardware evidence is the remaining reference and transport controls;
  a working first backend and beating every mx gate are separate checkpoints.
- **Landing checks:** docs and dead-code pass on the integrated current-main
  tree, using the existing f80709f20 executable; no runtime source changes.
  All 102 tracked Markdown pages were inventoried with their local links.
  This landing adds the plan and pinned research, updates the roadmap link and
  prepends this evidence record while preserving main's existing status text.
  The four changed pages are ASCII; their claims were checked against the
  retained sources, builds and raw runs. Unchanged pages retain the earlier
  review, including separately tracked stale tensor-split wording; historical
  measurements are not presented as freshly rerun model gates.
  The docs-only merge tier requires these two checks and does not wait for
  hosted CI. This consolidated plan lands as one commit by fast-forward.

## Vulkan int8 twin cache row width (2026-10-09, branch fix/vulkan-int8-twin-width, lands by fast-forward)

- **Goal:** reusing one activation buffer with a different matrix row width must
  give the same output as freshly packing that input for the new width.
- **Done:** the public-call regression on main `a1b8211b4` fails on an MI50:
  `int8 twin reused across matrix widths changes output`. It witnesses the
  int8 tile; 1,239,200 existing int8 outputs pass before the new case.
  The unfixed cache accepted any block-major width for a nonzero tile width,
  although addressing depends on it. The build and failed run are retained in
  `/zpool1/llmx-xdev-validation/quantizer-push-20261009/twin-width`.
  The cache owner now requires a matching block-major width while preserving
  reuse of column-major and 16-bit copies. The same MI50 test passes with
  5,120 new output values exact, alongside the existing backend checks.
  Both touched files' comments are swept: 55 blocks shortened with all
  non-comment tokens retained, and the detailed decode/count notes moved to
  the Vulkan source page. Integration with the comment checker removes
  those two files' obsolete allowances and adds no exception.
  A fresh MSVC Vulkan build after that sweep passes
  `backend-vulkan` on the Radeon VII; the int8-specific case reports zero
  there because its profile does not select int8. The compile log records
  both edited files rebuilt. Qwen3-0.6B Q8_0 full-vocabulary logits agree byte
  for byte against the unchanged main runtime on CPU and Radeon VII, under
  both auto and int8 fallback (four comparisons). These quick comparisons
  cover one prompt, not the complete model or performance gates.
  Docs and dead-code checks pass; the docs check's initial misreading of a
  mathematical expression as a C++ call was corrected in prose.
  The complete fresh Windows build passes all 48 native tests in 564.37
  seconds, including the five Vulkan checks; logs are `build-all.log` and
  `ctest.log` in the branch worktree. Existing test-source shadowing and
  `sscanf` warnings remain; no warning-free build is claimed.
  A clean detached Linux build at f3dce7978 passes all 47 native tests on
  an MI50 in 594.09 seconds. All six required GGUF fixtures match their
  pinned sizes and SHA256 values; the full device suite with required tools
  and baseline passes all 28 components, including the HF, int8, MXFP4,
  qwen35, split, server and follow-up checks. These results are in
  `/zpool1/llmx-xdev-validation/int8-twin-gate-20261009/evidence`.
  All 72 model identity comparisons against the unchanged main runtime
  pass: six pinned Qwen3 and Qwen3.5 files, CPU and MI50, auto and int8,
  each with full-vocabulary prompt logits, eight fixed continuation tokens'
  full-vocabulary logits, and up to 32 generated token IDs and their text.
  The original generation comparator included timing lines and looked for
  IDs on stderr, so its 24 failure verdicts are invalid. Its raw captures
  are retained; `identity-verified.json` compares the actual IDs and text
  on stdout and records the added continuation checks. This preserves the
  existing arithmetic on these workloads; it is not a new depth gate.
  The matched timing round runs both clean builds on one MI50 (GPU4,
  PCI 0000:89:00.0), default clocks, two host threads pinned to logical
  CPUs 0 and 1 with their siblings reserved. Each model/dtype runs
  main/fixed/fixed/main, pp512/tg128, three repetitions per command,
  ubatch512 and F16 K/V. All 48 calls are retained; the table gives the mean
  of each arm's two calls in tok/s. This repair's comparison is against
  main; the phase-level mx gaps remain open and mx was not remeasured here.

  | Model | Dtype | Main pp | Fixed pp | Change | Main tg | Fixed tg | Change | mx |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | Qwen3-0.6B Q8_0 | auto | 8778.60 | 8785.51 | +0.08% | 402.10 | 401.00 | -0.27% | Not remeasured |
  | Qwen3-0.6B Q8_0 | int8 | 13431.10 | 13491.74 | +0.45% | 394.33 | 394.77 | +0.11% | Not remeasured |
  | Qwen3-0.6B Q4_0 | auto | 8272.35 | 8267.84 | -0.05% | 348.18 | 349.43 | +0.36% | Not remeasured |
  | Qwen3-0.6B Q4_0 | int8 | 12277.63 | 12274.39 | -0.03% | 391.81 | 394.38 | +0.66% | Not remeasured |
  | Qwen3-0.6B Q5_K_M | auto | 8060.59 | 8057.66 | -0.04% | 353.25 | 355.87 | +0.74% | Not remeasured |
  | Qwen3-0.6B Q5_K_M | int8 | 11711.56 | 11702.06 | -0.08% | 366.14 | 366.88 | +0.20% | Not remeasured |
  | Qwen3-0.6B Q4_K_M | auto | 8143.90 | 8138.42 | -0.07% | 361.49 | 360.01 | -0.41% | Not remeasured |
  | Qwen3-0.6B Q4_K_M | int8 | 11883.54 | 11891.47 | +0.07% | 376.49 | 376.83 | +0.09% | Not remeasured |
  | Qwen3.5-0.8B Q8_0 | auto | 7517.61 | 7520.21 | +0.03% | 346.50 | 349.51 | +0.87% | Not remeasured |
  | Qwen3.5-0.8B Q8_0 | int8 | 11064.90 | 10840.67 | -2.03% | 347.46 | 346.56 | -0.26% | Not remeasured |
  | Qwen3.5-0.8B Q4_K_M | auto | 7023.35 | 7016.27 | -0.10% | 313.70 | 313.61 | -0.03% | Not remeasured |
  | Qwen3.5-0.8B Q4_K_M | int8 | 9894.28 | 9903.64 | +0.09% | 323.18 | 327.99 | +1.49% | Not remeasured |

  The largest prefill loss comes from one Qwen3.5 Q8_0 int8 fixed call at
  10608.97; the other is 11072.36 against main's 11077.40 and 11052.39.
  Two further ABBA blocks, declared before running, retain that workload
  and all flags. They give main/fixed 11088.86/11055.35 prefill (-0.30%)
  and 346.06/343.90 decode (-0.62%). Retaining all six calls per arm,
  including the initial slow one, gives -0.88% prefill and -0.50% decode.
  The initial two-percent loss does not persist; its cause is unproven.
  No causal speedup or layout effect is claimed. The small observed cost
  is retained beside the cache correctness repair, not rejected or hidden.
  Every initial monitor interval flags unrelated CPU work on other cores;
  seven also flag disk activity. Process starts/exits and 68 inaccessible
  process events limit coverage, and no unavailable counter is called idle.
  The one-second monitor covers loading and execution together; the CLI
  retains each command's mean and standard deviation, not each repetition.
  Raw outputs, commands, binary hashes, versions, clocks and monitoring are
  retained beside the identity evidence in `timing-1`, `timing-activity-1`,
  `timing-confirm-1` and `timing-activity-confirm-1`. No run is discarded.
- **Review:** the checkpoint inventory covers all 100 tracked Markdown files
  and 496 local link targets, all present. The docs and dead-code checks
  pass, and the precision, usage, architecture, test coverage and Vulkan
  owner pages were checked against their code and the retained results.
  Older live claims that tensor split and the Vulkan collective are planned,
  and that the Vulkan output head always uses 16-bit inputs, were sent to
  the owner of the separate documentation sweep. Historical measurements
  retain their dates and scope; none is presented as a fresh reference run.
- **Landing:** two commits, the failing test then the correction, by fast-forward
  after integration onto current main, rebuilt native tests on both platforms
  and the exact-head hosted gate. A code-conflict-free rebase retains the
  model and timing evidence above; any code conflict requires a new assessment.
- **Left:** the existing phase-level performance gaps against mx remain open.
- **Gotchas:** the first regression narrows the input rows from 512 to 256 so
  the incorrect cache reads stay inside the larger existing packed allocation.
  It needs native int8 tile execution; the Radeon VII's default profile does
  not advertise that policy, so the path is checked manually on an MI50.

## Vulkan quantizer test push constants (2026-10-09, branch fix/vulkan-quantization-push, lands by fast-forward)

- **Goal:** make the word-wise activation quantizer checks set every push
  constant read by the shader, including its layout width.
- **Done:** rechecked the two test helpers against the shader and production
  dispatch. The old test sent only the count; the word shader also reads `major`.
  Added a regression that seeds a prior block-major layout before the existing
  plain-layout checks; its guard also contains that layout's extra table padding.
  A fresh MSVC/Vulkan build on the Radeon VII fails with
  `8-bit activation guard changed`; all 158076 16-bit lane/word words agree
  before that failure. Logs: `build-check/quantization-build.log` and
  `build-check/quantization-unfixed.log` in the isolated feature worktree.
  Both word-wise helpers now push the count and explicit plain layout. The
  corrected Radeon VII check passes: 87820 identical int8 lane/word words,
  50676 unchanged Q8 consumer outputs and all existing offset/range checks;
  the simulated no-preservation backend retains its documented subnormal skips.
  The compile log confirms the edited source rebuilt. Fixed logs are
  `build-check/quantization-fixed-build.log` and `quantization-fixed.log` there.
  The docs and dead-code components pass. The test's comments and its AGENTS
  coverage description were reviewed; runtime source and shaders are unchanged.
- **Gates:** all 48 native tests pass on the fresh Windows Release build,
  including the five Vulkan tests, in 282.15 seconds. The MI50 also reproduces
  the regression's guard failure and passes the correction at bf617912f; its
  isolated container is removed. Linux logs, source and binary hashes are under
  `/zpool1/llmx-xdev-validation/quantizer-push-20261009`.
- **Landing:** two commits, the failing test then the correction, by fast-forward
  after the exact-head hosted run succeeds. Runtime source and shaders are
  unchanged, so model numerical and performance gates are not repeated.
- **Left:** the separate int8 twin cache-key repair and its gates are recorded above.
- **Gotchas:** this fixes test setup, not runtime arithmetic. GPU checks are
  manual because the hosted runners have no device. Prior passing results with
  an undefined layout word do not prove the intended packing path ran.

## Status table

| Feature                                  | Status   |
|------------------------------------------|----------|
| Layered restructure                      | Done     |
| Build config (config.hpp + CMake + build.bat) | Done |
| Test suite (roundtrip / perf / tokenizer)| Done     |
| Native server wave submission synchronization | Done |
| Server consistency tool: fresh phases and matched cache selection | Done (main `83b943a`, seven hosted jobs passed) |
| Perf `bench` command                     | Done     |
| CPU backend optimization                 | Done     |
| Early backend weight-type refusal and per-layer stream fallback | Done (main `737e082`, six hosted jobs passed) |
| CPU tiny-activation range repair | Done; measured CLI Q5 decode cost retained in its record in [STATUS-2026-09](STATUS-2026-09.md) |
| Disk tier under the host tier (docs/DISK-TIER.md) | Done; steps 1 to 6 on main, measured in the record below; since 2026-10-08 its files are what each turn changed (record below), its fault cases and counters built (record below) |
| A message boundary where a request's last user message starts | Done at `109784d3` (record below) |
| Vulkan allocation failure ownership | Done |
| Vulkan attention width and mixed-cache validation | Done |
| More quant formats (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K read) | Done |
| Activation dtype: resolution, CPU/Vulkan conformance, reporting and fallback | Done; qualified F16 default, explicit F32 and BF16 emulation, with the gates and retained speed gaps in the 2026-10-02 record below |
| CPU fit reserve beyond payload buffers | Done; test-first repair, startup under the same 8 GiB limit and integrated server coverage recorded below |
| Quantization coverage: F16/BF16, MXFP4, IQ4, Q3_K, Q2_K | In progress (record in [STATUS-2026-09](STATUS-2026-09.md)): the spec decoders, fixtures and MXFP4 writer merged at `e9b13dec`; CPU MXFP4 and explicit capture shares merged at `d81ed428`; Vulkan MXFP4 landed with the activation dtype (its row above). Remaining weight types and the recorded speed gaps stay open |
| More model architectures (Llama, ...)    | Planned  |
| Qwen 3.5, 3.6 and 3.8 (`qwen35`, `qwen35moe`) | In progress (record in [STATUS-2026-09](STATUS-2026-09.md), design in [QWEN35](QWEN35.md)), built in the background; step 4's references and CPU ops merged at `a730810`, and its model, which runs dense qwen35 on the CPU, merged at `c348cfb0`; step 5, the Vulkan backend, merged at `8d68d529` |
| Architecture modules: one runtime, a module per architecture, one registry | Done: merged at `3e73ffb` (record in [STATUS-2026-09](STATUS-2026-09.md)); the CPU timing on a quiet host follows |
| More formats (safetensors, ...)          | Planned  |
| JSON syntax and Unicode validation      | Done |
| GGUF reader size and tensor extent validation | Done |
| Checked conversion output publication | Done: merged at `4b0ec6a5`; checked write/close before replacement |
| JSON quantize tensor validation | Done |
| Q4_0 finite-input packing | Done: merged at `99572401`; tiny-scale reciprocal overflow handled |
| Qwen model construction validation | Done |
| Paged KV cache (block pool, backend-owned blocks) | Done |
| Device execution model (ROADMAP #4a)     | Done     |
| Execution model: tickets, batched views, placement (`docs/EXECUTION.md`) | Done: steps 1 to 7, step 7 being the server, see the server row; `--device` lists select a layer split (multi-device row) |
| KV cache fork (KV-CACHE step 2)          | Done     |
| Multi-device split (per-layer, per-tensor) | In progress (`docs/MULTI-DEVICE.md`): phase 0 measured, phase 1 (the layer split over a `--device` list fitted to free memory) and phase 2 (a prompt pipelined over the stages) merged; phase 3, passes in flight: step 1, the pass API, step 2, the scheduler over it, step 3, a pass in flight per stage and the 16-slot command ring (`ec03dcfa`), and step 5, the wider Q8_0 decode builds (`perf/decode-columns`), merged; step 4, the in-place rows and the sampling pool, merged at `41b19afc`; step 6, the head split, measured on 2, 3, 4 and 8 cards and not merged (the user, 2026-09-29): bit-identical to one card, it served 17 to 23 percent below the whole head on 8 cards and below it on 2 to 4, and `feat/split-head` keeps it as a record to revisit after step 7; step 7 and the final server gate follow; the tensor split (`docs/TENSOR-SPLIT.md`): step 0 measured, step 1 (the shards) and step 2 (groups on the CPU) merged, step 3 (the Vulkan collective) merged with its speed gate open below the reference (2026-10-04), step 4 (tensor groups serving) merged, step 5 (the hybrid qwen35 models on a group), step 6 (the embedded drafter on a group), step 8 (routed experts on a group) and a thread for each member but the first at a sum merged (records below) |
| GPU backends (Vulkan first to write, ROCm first-class) | Vulkan implemented and the recorded dense-model device gate passed on both platforms (forty-seventh checkpoint, in [STATUS-2026-09](STATUS-2026-09.md)): Radeon VII decode 102-115% and prefill 109-455% of the same-card reference Vulkan build; one MI50 decode 102-115% and prefill 102-267%. These are dated gate results, not new measurements from this documentation review. ROCm planned |
| Multi-node / cluster                     | Planned  |
| Two-row decode builds for the Q4 and K-quant rows | Done: merged at `b5cc467a` (record in [STATUS-2026-09](STATUS-2026-09.md)) |
| Q8_0 decode by two-wide 16-bit dots | Done (record below): bit-identical, 10 to 21 percent at 3 to 9 sequences on an MI50; lands by fast-forward |
| A lone prompt read by every stage (phase 4) | Done (record below) |
| The 16-bit prompt tile on the MI50 (option C step 1) | Measured, no gain in the shader alone (record below); the repacked Q8_0 layout recorded as the lever |
| `--dtype int8` at every row count (option C step 2) | Done at `c8b2ac8d` (record below) |
| The attention tile addresses its staged words directly | Done (record below): bit-identical, 10.5 percent of the tile on an MI50; lands by fast-forward |
| The 8-bit twin block-major where the tile reads it | Done (record below): bit-identical, int8 prompts 1 percent faster on Q8_0 and 3 on Q4_K_M on an MI50; lands by fast-forward |
| Qwen3-8B Q8_0 int8 prompts against the reference fork | Open (record below): level at pp512, 7 and 13 percent behind at pp2048 and pp4096, which is attention; the 8-bit tile shape not built |
| Tensor split against the reference, same topology (record below) | Open speed cells at `--dtype int8` on Qwen3-32B Q8_0 and MI50s, each shape against the reference's own, figures in the record `Llmx against the reference in the same topology`: one user in every tensor shape, and a group of four at 1 and 4 users (the newest session, record "Drafting depth" below: 31.0 and 30.2 tok/s at one user on a group of four against the reference's 43.5 and 43.7, and 27.8 and 28.4 on a group of two against 30.5 and 30.6), recovery a backend whose submissions do not go through the kernel per sum (the planned ROCm backend, after the tensor split is complete on Vulkan; the kernel route is closed); a group of two at 64 users, recovery not yet named; inter-token p99 at 32 users on short prompts, recovery assembly by predicted stage time (phase 3, step 9). The hybrid models and the embedded drafter have open cells of their own in their records. The K-quant mixture-of-experts files run under a tensor width with covering blocks and have open cells of their own at int8, in the routed experts' record. |
| Grouped `/v1/health`, `/v1/live`, grouped help pages and `docs/OPERATING.md` | Done (record below): one shape, no copy of the flat fields; lands by fast-forward |
| Multi-user server                        | Done (`docs/SERVER.md` steps 1 to 12 merged, 13 and 14 on `feat/split-sampling`; later split work is tracked in the multi-device row): `llmx serve`, correctness gates pass on both backends, throughput on one MI50 with Qwen3-8B Q8_0 132 and 174 percent of the reference server at 1 and 16 users and 85 percent at 4, in phase 3 step 2's gate (short of the wide margin `docs/SERVER.md` gates on), prefix reuse through fork, a second execution context measured and not added, since the next pass's tokens come from the one before, the OpenAI-compatible routes |
| Chat follow-up cache validation          | Done |
| Correctness baseline vs HF reference     | In Progress |
| Pinned HF reference generation           | Done |
| Optional Qwen3-8B HF consumer             | Done |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Matched CPU comparison thread selection | Done |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | Done |
| CPU attention in backend (ROADMAP #4a)  | Done |
| CPU row streaming / parallel prefill   | Done |
| CPU attention value accumulation      | Done |
| CPU grouped projections              | Done |
| CPU Q8 activation precision and batched float decode | Done (`36350ca`) |
| MI50 prompt activations at 16 bits | Done (`ef3b9428`); its speed gate stays open, the accumulating dots merged at `5bb4596f` and five prompt cells left ([STATUS-2026-09](STATUS-2026-09.md), MI50 prompt speed at 16 bits) |
| CPU Q8 scale / load scheduling       | Done |
| Head-major CPU KV storage             | Done |
| CPU worker exception safety           | Done |
| CPU worker cost profile                 | Done |
| CPU ordered prefill reductions          | Done |
| Backend-owned prefill placement | Done (main `3c5d4b9`, five hosted jobs green) |
| CLI thread settings, including batched/per-token perplexity | Done |
| Automatic build identification          | Done (main `9511a4a`) |
| Focused CLI help and complete current option coverage | Done (2026-09-24 checkpoint) |
| Live generation and loading progress     | Done |
| Model loading and teardown buffer lifetime | Done (2026-09-25 checkpoint) |
| CPU zero-byte transfer and zero-thread hint contracts | Done (2026-09-25 checkpoint) |
| GitHub CPU CI                          | Done     |
| HF fixture download retries and CI cache | Done |
| Hosted numeric/path portability repair | Done (five jobs green at `851d375`) |
| HF model download and sharded GGUF (ROADMAP #9a) | Done (included in main; five hosted jobs passed at `7e195ff`) |
| HF native formats (ROADMAP #9b)          | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |
| Documentation consistency review | Done (merged at `5869385b`, six hosted jobs passed) |
| Shared device-versus-CPU numerical gate | Done (main `c6a91bf`) |
| Required device types in the test runner | Done (main `56172ea`) |
| Vulkan finite activation repair | On main at `99020607`; its block records the exact-head hosted run as still owed |
| Dead-code and stale-docs checks in every job | Done (merged at `75450ea`, record in [STATUS-2026-09](STATUS-2026-09.md)); the listed findings cleaned up on `cleanup/known-findings` (2026-10-06, its record below) |

`Done` denotes implemented and validated functionality in this release tree.
The earlier runtime base `08351b0` was published on both main remotes. Its initial five-check
hosted run `35512421834` passed ordinary Ubuntu and required HF, but failed
Windows reference-generator path spelling, UBSan exact scalar-tail comparison,
and macOS JSON subnormal conversion. Repair `851d375` passed all five jobs in
[run 35512954742](https://github.com/mxxm-t/llmx/actions/runs/35512954742):
Windows, macOS Intel, Linux, Linux UBSan and required HF. The repair changes
JSON conversion plus test portability, preserving inference kernels and bounds.
Evidence is in `docs/benchmarks/ci-portability-20260920.json`; prior four-check
passes at `b266650` and `9511a4a` cover those smaller releases.
See [CI](CI.md) for the precise workflow scope and local reproduction commands.

## Active feature blocks

### Scoped correctness coverage and remaining HF work

- **Goal:** keep independent HF ground truth and extend coverage where the roadmap requires it.
- **Done:** exact tokenizer fixtures; tiny tied/untied F32 full logits and NLL; real Q8_0, Q4_0, Q5_K_M and Q4_K_M ranking and excerpt PPL; HF/Jinja2 follow-up chat fixtures; pinned reference generation and strict consumers. The real 8B consumer, now 41 checks with each NLL case scored batched and per token, passed 41 of 41 on one MI50 at `a2b732f` and in both Q8_0 decode orders at the half-block order's merge; its earlier 37-check form passed 37/37 on Windows and Linux with frozen bounds. Real 0.6B F32/Q8 1,943-token plus 32-step continuation checks are archived in ASSETS.
- **Left:** broader full-corpus, maximum-context and per-layer references, plus prospective numerical bounds for any new lossy kernels. Short 8B rankings/excerpts are not deep-context validation.
- **Gotchas:** self-consistency is supplementary. Exact comparison against another llmx path cannot replace HF. Model construction validation does not establish finite weights, arbitrary token-ID safety, request budgets or failed-session recovery.

### The tensor split on every model the layer split serves

- **Goal:** `--tensor-width N` runs each family the layer split runs, with its gates: the hybrid qwen35 models (Qwen 3.5, 3.6 and 3.8), then an embedded drafter over a group, then routed experts (qwen3moe, qwen35moe); the plan's steps 5, 6 and 8 (`docs/TENSOR-SPLIT.md`, section 6).
- **Done:** the hybrid models (branch feat/tp-qwen35, its record below): a linear-attention layer on each member over its K heads and the V heads that read them, each member keeping the recurrent state of those heads under the slot ids the group shares, so checkpoints, marks, the rerun after a retract, the host tier's copies and drafting by lookup run over a group. The embedded drafter (branch feat/tp-spec, its record below): each member its shards of the MTP block and the head whole for a draft row, with no join of the members' argmax. Routed experts (branch feat/tp-moe, its record below): each member its share of every expert's hidden rows beside the whole router. Covering blocks (branch feat/tp-moe-cover, its record below): the K-quant files whose expert width does not split on whole blocks.
- **Left:** the join of the members' argmax in `Collective` if a draft row's head shows in a profile; the open speed cells of each record.
- **Gotchas:** a member's activation slots keep one device's widths, so its arena is larger than it needs by the share it does not use; the fit counts it as allocated.

## Working rules and ownership

Current feature ownership and timing reservations are recorded in the shared
collaboration log outside this repository. Confirm ownership there before
starting work; historical branch names below are not active assignments. Builds and tests
may run in parallel when no timing reservation is active. Keep every planned
performance sample, record ordinary machine activity, and report missing
telemetry honestly. GitHub receives main and the `gate/<name>` branches whose hosted run checks a stack before it merges (`docs/CI.md`); feature work stays on its branch until its gates pass.

## Records (2026-10-01 to 2026-10-08)

Each dated block below is the record of a change as it landed or was measured, newest first: what was found, what was done, what the gates measured and what it left open.
The status table and the active blocks above give the present state; a record's open items may have shipped since.

## A prompt that does not fit its context is refused with its numbers or cut at a turn's end (2026-10-08, branch feat/context-overflow, lands by fast-forward)

- **Why (the user, 2026-10-08):** a server at its context limit must stay usable; a conversation that outgrows the context got a 413 on every turn from then on. "Have an option to refuse and cut", with cheap markers for where to cut; the default stays `refuse`.
- **Done:** `serve --context-overflow refuse|shift`, one rule a value on all four generating routes. `refuse`, the default, answers 413 with the prompt's tokens, the limit and the flag's name. `shift` cuts the prompt before the scheduler sees it: a chat route drops the oldest messages after the system message, whole messages, a kept window starting at a user message so a tool call stays with its results, and renders the rest again; a text route keeps its leading block and drops whole turns after it, or whole steps of rows where the prompt has no end token. Every reply carries a `context` object beside `usage`: `tokens`, `limit`, and after a cut `dropped_messages`, `dropped_tokens` and `at_marker`.
- **The index** is one scan of the prompt's ids for the tokens the tokenizer's metadata marks as ends, a row offset a turn (`src/server/context_cut.hpp`, new, 74 lines, with no model, tokenizer or template in it); the chat routes fall back to rendering the conversation up to each message where the scan finds another number of ends than there are messages. `api.hpp` goes from 839 to 910 lines; `scheduler.hpp` is untouched.
- **Steps from the prompt's start, not a target from its end:** a cut drops to the first cut point one step, half the limit, past the leading block, or two steps, the fewest that fit. The first form, "drop until half is left", moved the cut at every turn, since every turn's conversation is a little longer, and so read the window again every turn; counted from the start the cut stays at the same message for a step's worth of conversation. On the real model's test, Qwen3-0.6B Q8_0 on a pool of 2048 tokens, the cuts over fourteen turns are none for eight turns and then 10 messages on each of the six after, and each turn cut where the one before was reuses it.
- **What a cut costs, and what no marker saves:** the turn that cuts reads everything past the leading block again, on every model; a saved state holds the dropped messages, and without one the kept rows' positions change. At one half each new row of conversation costs about one row read again. On Qwen3.8-27B Q8_0 at 262144 tokens that is a wait of up to about eight minutes at a cut, once every 131k tokens of conversation; reading the shifted window ahead while idle is the measured second step and is not built.
- **The failing test first:** the suite's `server` component: on main the server does not take the flag; with the change the synthetic chat model on a pool of 128 tokens refuses by default on the four routes with the numbers, and under `shift` a chat of six turns is cut by 0, 0, 4, 4, 6 and 8 messages, each reply and prompt those of the kept messages sent alone, a text prompt by one step after its first rows; the Q8_0 fixture on 2048 tokens as above, a pool that large so that the reuse is held on a device too, where a window under 449 tokens takes another tile split at every length. `context-cut` (new CTest) holds the cut alone.
- **Not built:** the idle read of a shifted window; a cut that could keep part of a turn.
- **Measured, gates:** on Windows CTest 43 of 43 and the whole CPU suite; the test machine's CPU gates, the suite's server component on an MI50, identity of Qwen3-0.6B Q8_0 with main and the hosted run are named in the landing's devlog entry.
- **Reviewed:** by F2DEV, no finding in the code: the steps counted from the prompt's start taken as an improvement on the note, the estimate checked by the real render before anything is served; two notes taken, a clause on the fallback's cost in `message_ends` and a sentence of SERVER.

## A request drafts while another waits for room (2026-10-08, branch perf/draft-while-waiting, lands by fast-forward)

- **Found on production:** three requests generated 7156 tokens with at most 34 verifies between them. A pass took no draft while any request was queued or paused, a rule no record names a measurement for, so the request that held the pool ran at its plain rate for exactly as long as another waited for it to end.
- **Measured first** (two MI50s, Qwen3.8-27B Q8_0 with its embedded drafter, production's flags and a 16384-token context, a chat generating to the pool's end with a second one paused beside it): from the pause to the first request's end, 160 s and about 3300 tokens, the verifies stood still at 253, and the request ran at 20.8 tok/s.
- **Done:** a request paused or queued for room no longer stops the drafts; it waits for the running requests to end or grow, which their drafts bring sooner, and a draft stays inside the drafting request's own reservation. The rule stays where every seat is taken and a request is queued for one, which is the half a measurement asked for (below). `scheduler.hpp` is 2769 lines before and 2771 after.
- **The same reproduction with the change:** the first request runs at 31.3 tok/s and ends after 147 s where it took 220 s, 1328 verifies by then; the second, which waits for it, has its whole reply 71 s sooner.
- **Why half of the rule stays:** removed whole, it cost output where requests queue for seats. Qwen3.8-27B Q8_0 with its embedded drafter, 16 seats, closed loops of 32 and 64 users of 128-token prompts and replies, tok/s of output, two rounds interleaved with main:

  | shape | users | main | the rule removed whole | the half kept |
  |---|---|---|---|---|
  | tensor group of two | 32 | 107.9, 106.2 and 106.0, 104.0 | 105.2, 106.3 | 105.4, 104.6 |
  | tensor group of two | 64 | 107.3, 107.3 and 107.5, 107.3 | 107.2, 107.5 | 107.7, 106.2 |
  | layer split of three | 32 | 136.8, 136.5 and 133.0, 136.2 | 122.4, 133.2 | 131.9, 130.3 |
  | layer split of three | 64 | 140.4, 140.4 and 136.9, 138.5 | 127.6, 134.4 | 139.0, 136.2 |

  Main was run in both sessions, so its cells hold four figures. On the split of three the whole removal lost 3 to 10 percent of the output and raised the first-token p99 from 48.4 to 51.0 and 54.6 s at 64 users, the passes feeding 488 and 987 drafts where main fed 130 and 165; with the half kept the cells are within main's own spread but for 32 users on the split, 131.9 and 130.3 against 133.0 to 136.8, which I leave stated and not explained. With 64 seats, where nothing queues, both builds are level (group 110.5, 123.9, 126.8 against 110.8, 125.4, 127.5 at 16, 32 and 64 users; split 132.9, 154.5, 155.5 against 135.6, 154.6, 155.7).
- **What the split's loss says and this branch does not fix:** drafts beside 16 decoders on a layer split of three cost output, and the price that should refuse them takes them. That is the price's defect, with the third draft it leaves on a group of two (the record of the drafting rounds); the kept half of the rule covers it only where requests queue for seats.
- **The failing test first:** `server-spec`: on two seats and 8 blocks a request with drafts beside a queued request the pool has no room for must feed the drafts it feeds alone, which on main it does not, feeding none of them; on one seat, with a second request queued for it, it feeds none.
- **Measured, gates:** the two tables above; on Windows CTest and the suite's server, qwen35, docs, dead-code and version components; the test machine's CPU gates with the linked check, identity of Qwen3-0.6B Q8_0 with main and the hosted run are named in the landing's devlog entry.
- **Reviewed:** by F2DEV, no finding: the rule narrowed to its measured half, its reason beside it; the price's taking drafts beside many decoders on a layer split named as its own defect and left as one item with the third draft on the group.

## `/v1/health`'s host figures say what they count (2026-10-08, branch docs/health-host-wording, docs only, lands by fast-forward)

- **Why:** production showed `reuse.host.now.entries` 0 beside 536870912 `bytes`, and 313868288 `bytes_moved` with no promotion, which read as an error and is not one.
- **Done:** USAGE's table says that `entries` counts copies of histories alone, that `bytes` is the slabs the tier holds, whole slabs, for copies, for message boundaries' states and for histories read back from disk, that `bytes_moved` includes each boundary's state as it is kept, and that a boundary's use counts under `reuse.boundaries`. No source changes.
- **Traced** on main `a0922bb62`: `host_.size()`, `host_held_`, `host_hits_` and `host_moved_` in `src/server/scheduler.hpp`, the boundaries adding to the held and moved bytes where their state is kept. `timing.stage_idle_share`, asked about with these, is written only under `--timing`, as USAGE and SERVER say.

## Covering blocks: the K-quant mixture-of-experts files under a tensor width (2026-10-08, branch feat/tp-moe-cover, lands by fast-forward)

- **Goal:** the files the routed experts' step refused: Qwen3-30B-A3B in a K-quant at every width and Qwen3.6-35B-A3B in a K-quant at width 4, an expert's share of its hidden columns not being whole 256-value blocks of its down stack.
- **Measured first** (Qwen3-30B-A3B Q4_K_M, one MI50, main 6f71114ca, `bench --model --p 512 --n 128 --r 2 --profile`): a decode token is 9.2 ms of device time over 640 dispatches, 40.5 percent in the Q4_K row kernel and 19.6 in the Q6_K one, and by the weights those read a layer the experts are two thirds of them, so expert products are about 40 percent of a decode token; a prompt of 512 is 87 percent in the three tiles. On Qwen3.6-35B-A3B Q4_K_M, which runs with even shares, a group of two reads a prompt 1.35 times as fast as one card and decodes below it (the routed experts' record, below), so the form that opens the K-quants should cost little and need nothing below the model.
- **The form, and the two not built:** `docs/TENSOR-SPLIT.md`, section 8, Covering blocks as built, with per file and width the columns a member owns and covers, its share of an expert's work and the experts' bytes over the group. Decided by the coordinator on the proposal of this date, read by F2DEV, whose three points are in: the zero rows written through the loader, zero blocks decoding to zeros pinned in `raw-blocks`, and those numbers in the record.
- **Done:** `ShardSection::align`; `shard::parts` (what a member owns and what it holds a tile), `spans` the covers, `runs` with `zero` runs after the file's, `bytes` and `pack` over both; `blocks::shard_experts` reads the down stack's block size from its tensor, and splits qwen35moe's shared expert as one more expert, which the first card run showed was refused at width 4 on Qwen3.6-35B-A3B Q4_K_M while declared as a dense block; `planning_adopt` writes a shard's zero runs as it places the storage and records the file's runs alone; the streamed loader's packed writes stop where the runs stop following one another, which the zero rows made possible and a CPU group's load of the new fixture caught at width 4. `check_plan` refuses no expert width; the refusal's line and case are gone. Nothing changed in a backend, the collective, the scheduler or the server, and a file whose shares are whole blocks holds what it held.
- **Correctness on CPU groups** (hosted): a Q8_0 qwen3moe fixture whose experts are three blocks wide, with HF goldens made from the file's own weights (`docs/ASSETS.md`, Tensor-split fixtures): at widths 2 and 4 within 1.4e-6 logits (bound 2e-5) and 1.2e-7 NLL (bound 1e-5) of HF and inside the device-reference criterion against one backend, its load streamed through the loader; as two stages of groups of two against one group, bit-identical (`split`); `shard`: the covers of a 768-wide expert in Q4_K, Q6_K and Q8_0 at widths 2 and 4 byte for byte, the file's rows where a member owns a column and zero rows elsewhere, each row owned once; `raw-blocks`: zero bytes decode to zeros in all 14 types; `model-validation`: a routed Q8_0 plan taken at width 4.
- **Correctness on cards** (MI50s under one root, the branch's code at ef36955c1, default precision): the tensor-split component on groups of two and four, the covering fixture among its six, within 1.04e-4 logits and 7.6e-6 NLL of HF, the device bounds, and inside the device-reference criterion against one card. Qwen3.6-35B-A3B Q4_K_M at width 4, each member covering one of an expert's two blocks and of the shared expert's, the K-quant cover's check against HF: `tests/baseline_layered.py --tensor-width 4`, 63 of 63 checks, its file-exact goldens' and the model's six rankings and four NLL cases batched and per token.
- **Qwen3-30B-A3B Q4_K_M has no HF reference in the tree**, so it is compared with one MI50 over the 247 tokens of the pinned excerpt, every row batched and per token, and 64 greedy steps (`llmx-model-logits`, the counters of `common.check_device_rows` and `check_device_greedy`), beside three controls with no cover: Qwen3.6-35B-A3B Q4_K_M at width 2, whose group is held to HF, and the same model in Q4_0 and in Q8_0 as groups of two, the Q8_0, which one MI50 does not hold, against a layer split of two, one device bit for bit:

  | file, group against one MI50 | path | max logit gap | mean NLL delta (bound 0.01) | top-1 outside a tie | top-5 outside a tie | greedy, 64 steps |
  |---|---|---:|---:|---:|---:|---:|
  | control, Qwen3.6-35B-A3B Q4_K_M, width 2 | batched | 0.82 | 7.3e-4 | 1 of 233 | 2 of 243 | 64 |
  | control | per token | 0.82 | 3.4e-4 | 0 of 231 | 2 of 243 | |
  | Qwen3-30B-A3B Q4_K_M, width 2 | batched | 1.00 | 9.2e-4 | 0 of 237 | 0 of 247 | 64 |
  | Qwen3-30B-A3B Q4_K_M, width 2 | per token | 1.69 | 3.5e-4 | 0 of 234 | 3 of 247 | |
  | Qwen3-30B-A3B Q4_K_M, width 4 | batched | 1.00 | 8.0e-4 | 0 of 237 | 0 of 247 | 64 |
  | Qwen3-30B-A3B Q4_K_M, width 4 | per token | 1.69 | 7.3e-4 | 0 of 234 | 3 of 247 | |
  | control, Qwen3-30B-A3B Q4_0, width 2, no cover | batched | 1.34 | 1.7e-4 | 0 of 234 | 0 of 247 | 45, a near tie at 12 |
  | control, Qwen3-30B-A3B Q4_0, width 2, no cover | per token | 1.81 | 1.3e-3 | 0 of 234 | 2 of 247 | |
  | control, Qwen3-30B-A3B Q8_0, width 2, no cover | batched | 1.08 | 5.1e-4 | 0 of 240 | 0 of 247 | 64 |
  | control, Qwen3-30B-A3B Q8_0, width 2, no cover | per token | 1.00 | 5.6e-4 | 0 of 241 | 0 of 247 | |

  These per-token figures are a routed group's against one card and not a pass of the criterion: its top-5 check forgives a miss only within 0.1 logits of the fifth value, a margin set for one device against the CPU, and it is not a gate for a routed group, whose members' sums differ from one card's by their order and, at a near tie of the router, choose another expert, which moves a row by whole logits. Qwen3-30B-A3B Q4_K_M misses it per token at 3 of 247 positions with a gap of 1.69 at both widths (positions 105, 199 and 211 at width 2 and 105, 192 and 211 at width 4), though its covers differ between the widths. The coverless controls show the same scale: the Q4_0 group misses at 2 of 247 (positions 85 and 145, neither of the K-quant's) with a gap of 1.81, the HF-held Qwen3.6-35B-A3B group at 2 of 243 with a top-1 beside them, and the Q8_0 group holds top-5 with gaps of 1.0. So the misses are those of a 4-bit routed group and the cover is cleared of them; a cover adds exact zeros to a member's sum. NLL, top-1 and greedy hold at both widths. This is not an HF check of the file, which stays open, its reference fitting no host here whole; the cover with a block both members hold, this file's at width 2, is held to HF by the Q8_0 fixture of three blocks and byte for byte by `shard`.
- **Unchanged where shares are whole blocks:** against the routed experts' head a0922bb62, the greedy ids, a prompt's top logits and an excerpt's perplexity are the same bytes for Qwen3-30B-A3B Q4_K_M on one MI50, Qwen3.6-35B-A3B Q4_K_M as a group of two and Qwen3.6-35B-A3B Q8_0 as a group of four; the group of two times the same, 2157 against 2153 tokens/s reading and 79.7 against 78.9 decoding.
- **Speed** (MI50s under one root, default clocks, `bench --model --p 512 --n 128 --r 3`, two interleaved rounds, the mean of the two; the reference is its ROCm build at `-sm tensor` on the same cards in the same session, `--dtype int8` llmx's matched arm; tokens/s):

  | Qwen3-30B-A3B Q4_K_M | prompt, 512 | decode, 128 |
  |---|---:|---:|
  | one MI50, int8 | 1607 | 136.1 |
  | layer split of two, int8 | 1553 | 127.5 |
  | group of two, int8 | 1934 | 73.9 |
  | group of two, default precision | 1455 | 75.6 |
  | reference, group of two | 1485 | 108 and 123 |
  | group of four, int8 | 1546 | 37.2 |
  | reference, group of four | 1475 | 126.8 |

  | Qwen3.6-35B-A3B Q4_K_M | prompt, 512 | decode, 128 |
  |---|---:|---:|
  | group of four, int8 | 1718 | 40.3 |
  | reference, group of four | 1398 | 95.9 |

  The first round's first two cells, the reference's group of two (decode 108 +- 13) and llmx's beside it, ran for about a minute beside a capture of mine on the same cards; both rounds are kept. The reference's groups of four fell back to its internal sum, as its log says.
- **Open speed cells, first support:** a prompt is read faster than the reference at every shape (1.30 times on the group of two of Qwen3-30B-A3B, 1.05 on its group of four, 1.23 on Qwen3.6-35B-A3B's group of four) and, on the group of two, 1.20 times as fast as one card. Decode is below the reference and below one card at every group: 0.64 of the reference on the group of two and 0.29 on the group of four of Qwen3-30B-A3B, 0.42 on Qwen3.6-35B-A3B's group of four. A group's decode is bound by its two sums a layer and the members' turns, the cells the tensor split's own record lists and its member-threads work recovers, not by the covers: the group of two of Qwen3.6-35B-A3B, with no cover, decodes at 0.58 of one card as this file's does at 0.54.
- **Left:** HF goldens of Qwen3-30B-A3B in a K-quant if that file is to be held to HF itself; the three owners and an idle member at its width 4 as a measurement; the open decode cells.
- **Reviewed by:** F2DEV: the code at bb3fee330 with no finding, the shared expert's delta at ef36955c1, and the card gate, with the ruling this record follows: it lands as first support once a coverless control clears the cover, which the Q4_0 control does, and the per-token figures are stated as a routed group's.
- **Gotchas:** the fixture is Q8_0, 32-value blocks, since the tests have no K-quant writer; the cover's code reads only the block size, and the K-quant path itself is the cards' check. A covered file costs memory, 4/3 to 2 times the experts' bytes over the group, which the fit counts from the shard.

## A resumed request is not asked for a draft before a pass has fed it, and a failed draft ends no request (2026-10-08, branch fix/draft-after-resume, lands by fast-forward)

- **Found on production** (the user: a server that breaks is not correct behaviour): a request of 199800 prompt tokens with 20105 generated ended "error" beside a second request, near the pool's end, the log line giving no reason.
- **Reproduced** on two MI50s, Qwen3.8-27B Q8_0 with production's flags and a 16384-token context, and then on the CPU with the tiny hybrid model: the error is `inference: a draft of a history no pass has fed`, and the pool running out is only how the server gets there.
- **The path:** an uncapped request that cannot grow sits passes out at its reservation's end, a whole block, and is paused for an older request's growth; its parked history goes to host memory as that request grows on. When room returns it resumes by a fork of the copy promoted from host memory, which holds its whole history, so it decodes at once, on a sequence no pass has fed since the fork. On a model that keeps a state such a sequence has no live state yet, the scheduler asked the embedded drafter for a draft of it, the model threw, and the round's catch ended every active request with the error.
- **Done:** `propose` leaves out a request whose sequence no pass has fed (`Model::can_draft`), so its first pass after a fork is a plain decode entry; a draft that throws costs its pass the drafts and no request, is logged with its reason and counted (`drafting.since_start.failed` in `/v1/health`); the request line gives an error's reason after the word. `scheduler.hpp` gains 11 lines, 2760 to 2771.
- **The failing test first:** the suite's `qwen35` component serves the tiny hybrid model with its drafter and a host tier and runs the two requests; on main the second ends with the error, three runs of three on Windows, and with the fix both end by length with their texts alone. `server-spec` gains the same path through the scheduler, which passes on main too, the drafter resting at that moment there, and holds the replies to the requests alone.
- **Not this branch:** the two rules for the cap itself, room made by dropping what no running request needs and a sliding window for a conversation larger than the context, are designed next; this defect was under both.
- **Not proven here:** a draft that fails midway through a chain on a device, as against at the model's own checks, which change nothing; the catch keeps the request, and a pass that then fails on that device ends as any failed pass does, its own requests alone.
- **Measured, gates:** on Windows, CTest 42 of 42 and the suite's `qwen35`, `server`, `docs`, `dead-code`, `version` and `arch-boundary`; `qwen35` against a build without the fix fails with the production text in two runs of two, and passes three of three with it. On the CPU with the tiny hybrid model, 5800 and 1200 prompt tokens on a pool of 8192: before, the second request ends with the error after 337 tokens; after, both end by length, the second with 6992 tokens, one pause, nothing recomputed. On the test machine: a CPU build, CTest, the whole CPU suite and the linked check; and by hand on the cards before it landed, their figures in the landing's devlog entry: the two-card reproduction on Qwen3.8-27B Q8_0 at the fix, identity with main of Qwen3-0.6B Q8_0 and Qwen3.5-0.8B Q8_0 on the CPU and one MI50, the suite on the MI50 and one timing round. The hosted run at the landing head.
- **Reviewed:** by F2DEV: the change is the pair it asked for, the rule in the model and one call from `propose`, the catch returning before any mark is taken; one fix of placement taken, `can_draft` beside `mark` and `keep`; the device-failure path of a chain marked as not proven, above.

## Drafting depth, the rollback's cost, the tensor split's open cells again and the layer split's loaded cells rerun (2026-10-08, measured, docs only)

Measurements only: no source changed. Main is a0922bb62 (`llmx 0.1.0+ga0922bb623c8`), built once as a Release build with the Vulkan backend in the development image and used by every llmx arm. Raw outputs are in `docs/benchmarks/drafting-depth-20261008/`.
Every table was taken in one session on MI50s at default clocks, with the server or bench process on whole cores (a core and its sibling thread) that no other session used according to the shared log, arms interleaved or run from fresh servers in the order stated, and every run kept. A monitor wrote the machine's load average and the other sessions' containers every 10 seconds (`monitor.txt`); other sessions' work ran on other cores throughout, so the load average does not describe the cores of these runs. Its range per window is given with each table.

### Set 1: draft depth (`--draft-max`)

Qwen3.8-27B Q8_0, `serve --max-seqs 1 --ctx-size 8192 --drafter embedded --draft-max N` for N of 3 to 8 beside `--drafter off`, a fresh server for each arm, three requests a cell with the median shown (the server's own decode rate) and the drafts the server kept of those it fed over the three. Replies of 96 and 512 tokens with the end token ignored, greedy and seeded (temperature 0.8, top-k 40, top-p 0.95, seed 7). Prompt 1 is the 1500-byte extract of the earlier drafting record. Prompt 2 asks the model to copy a passage of about 900 bytes word for word, where drafts are easy; both prompts ran in the same server in that order. Shapes: one MI50, a layer split of two (`--device vulkan:0,vulkan:1`, production's shape), a group of two (`--tensor-width 2`) and a group of four (`--tensor-width 4`), each at default precision and at `--dtype int8`. The group shapes ran because the drafter now runs under a tensor width. The server drafts at a depth its pass price allows, so the fed counts fall below what a larger depth could feed.

Load average over the windows: one MI50, layer split and group of two 5.0 to 107.2 (13:15 to 15:32 UTC, other sessions' builds on other cores); group of four 7.4 to 37.7. Two group-of-four arms, int8 at depth 3 and depth 4, ran while another session loaded a file onto two of the group's cards for about a minute (16:01 UTC, the monitor shows its container from 16:01:24): their figures are marked affected below, and both arms were rerun in full (16:25 to 16:30 UTC, load 18.6 to 40.8, other sessions' work on other cores). The group-of-four int8 columns for depth 3 and 4 in the tables are the reruns.

The group of four's `off` cells at default precision vary from 17.8 to 31.8 tok/s between fresh servers (26.9, 17.9, 18.1, 20.6, 17.8, 20.9, 31.8 and 28.9 over the eight cells); its int8 `off` cells are 29.5 to 31.2.

Each cell is tok/s (drafts kept/fed).

one MI50, default: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 23.3 | 34.5 (166/341) | 36.1 (157/269) | 28.8 (103/175) | 28.9 (104/171) | 28.6 (104/168) | 26.7 (103/156) |
| 1, greedy, 512 | 23.5 | 52.8 (1066/1260) | 46.2 (1038/1200) | 45.3 (1017/1201) | 44.6 (1028/1238) | 44.8 (1033/1260) | 46.5 (1010/1170) |
| 1, seeded, 96 | 23.2 | 42.5 (181/303) | 37.8 (172/292) | 37.3 (170/295) | 34.7 (176/327) | 37.9 (183/356) | 42.5 (180/281) |
| 1, seeded, 512 | 23.3 | 49.0 (1054/1386) | 44.5 (993/1232) | 43.8 (965/1207) | 43.4 (1030/1333) | 43.7 (969/1242) | 44.7 (981/1240) |
| 2, greedy, 96 | 23.2 | 59.6 (211/211) | 58.6 (225/231) | 64.8 (237/240) | 66.1 (243/243) | 66.1 (246/252) | 48.6 (191/191) |
| 2, greedy, 512 | 23.5 | 55.7 (1104/1236) | 52.2 (1187/1353) | 55.7 (1230/1476) | 54.0 (1263/1569) | 56.5 (1287/1653) | 46.1 (1001/1091) |
| 2, seeded, 96 | 23.6 | 55.5 (204/234) | 51.1 (216/264) | 53.6 (225/282) | 52.3 (231/297) | 51.1 (234/318) | 45.9 (185/203) |
| 2, seeded, 512 | 23.5 | 45.8 (997/1458) | 42.9 (1055/1586) | 40.9 (1056/1766) | 42.4 (1055/1751) | 38.5 (1044/1888) | 42.3 (1001/1448) |

one MI50, --dtype int8: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 22.5 | 39.3 (167/342) | 39.1 (177/374) | 35.5 (181/505) | 37.8 (159/306) | 38.8 (157/279) | 30.4 (107/191) |
| 1, greedy, 512 | 22.6 | 56.1 (1089/1285) | 56.6 (1070/1264) | 56.7 (1129/1416) | 57.4 (1108/1359) | 55.0 (1136/1451) | 46.8 (1008/1262) |
| 1, seeded, 96 | 22.9 | 32.6 (137/378) | 32.3 (135/373) | 31.3 (130/345) | 32.3 (138/409) | 30.7 (141/479) | 29.7 (132/401) |
| 1, seeded, 512 | 22.7 | 35.0 (762/1603) | 35.1 (744/1505) | 34.3 (743/1465) | 34.9 (757/1581) | 33.7 (734/1533) | 33.7 (727/1435) |
| 2, greedy, 96 | 22.6 | 66.1 (213/213) | 68.0 (225/231) | 67.1 (225/231) | 85.9 (243/243) | 84.7 (246/252) | 77.9 (249/264) |
| 2, greedy, 512 | 22.8 | 62.1 (1125/1189) | 67.5 (1208/1284) | 67.0 (1211/1292) | 78.6 (1293/1419) | 82.1 (1320/1467) | 74.4 (1341/1497) |
| 2, seeded, 96 | 23.5 | 58.5 (204/234) | 59.7 (216/264) | 59.9 (216/264) | 67.1 (231/297) | 68.0 (234/318) | 61.6 (237/330) |
| 2, seeded, 512 | 22.7 | 52.2 (1056/1384) | 53.8 (1128/1588) | 55.4 (1155/1707) | 56.1 (1184/1941) | 55.9 (1205/2096) | 51.2 (1111/1728) |

layer split of two, default: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 22.8 | 37.5 (167/342) | 35.3 (157/273) | 28.9 (104/182) | 28.9 (104/171) | 28.8 (104/168) | 28.7 (100/152) |
| 1, greedy, 512 | 23.2 | 54.3 (1090/1282) | 52.5 (1071/1240) | 46.4 (1007/1181) | 45.7 (1025/1223) | 54.1 (1106/1265) | 47.1 (1009/1155) |
| 1, seeded, 96 | 23.3 | 43.8 (183/297) | 38.1 (171/291) | 38.3 (169/286) | 38.2 (177/327) | 38.1 (179/325) | 40.2 (167/260) |
| 1, seeded, 512 | 23.2 | 50.1 (1054/1386) | 44.9 (995/1222) | 44.8 (1005/1244) | 44.2 (998/1252) | 45.2 (1024/1358) | 44.3 (962/1171) |
| 2, greedy, 96 | 23.3 | 58.9 (211/211) | 58.7 (225/231) | 65.6 (237/240) | 69.9 (243/243) | 67.5 (246/252) | 50.1 (191/191) |
| 2, greedy, 512 | 23.2 | 56.0 (1104/1236) | 54.5 (1188/1349) | 56.4 (1230/1476) | 57.8 (1263/1569) | 57.1 (1287/1653) | 47.9 (1003/1087) |
| 2, seeded, 96 | 23.4 | 55.5 (204/234) | 51.2 (216/264) | 54.0 (225/282) | 55.6 (231/297) | 53.7 (234/318) | 47.1 (183/204) |
| 2, seeded, 512 | 23.3 | 45.9 (997/1458) | 42.4 (1039/1570) | 41.4 (1058/1755) | 41.6 (1086/1897) | 42.5 (1072/1861) | 44.6 (966/1377) |

layer split of two, --dtype int8: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 22.7 | 40.6 (167/342) | 39.7 (178/376) | 36.3 (181/506) | 38.8 (158/291) | 38.9 (157/278) | 39.4 (152/296) |
| 1, greedy, 512 | 23.1 | 58.0 (1089/1285) | 59.3 (1078/1291) | 58.7 (1124/1376) | 57.1 (1125/1428) | 56.6 (1137/1447) | 50.1 (990/1157) |
| 1, seeded, 96 | 23.1 | 33.5 (137/378) | 32.3 (129/334) | 33.3 (135/375) | 31.3 (131/404) | 31.6 (139/460) | 32.8 (126/285) |
| 1, seeded, 512 | 23.1 | 35.3 (749/1582) | 35.2 (726/1398) | 35.1 (728/1436) | 35.2 (746/1569) | 34.9 (732/1501) | 35.4 (723/1346) |
| 2, greedy, 96 | 23.1 | 66.6 (213/213) | 70.0 (225/231) | 79.3 (237/240) | 86.2 (243/243) | 88.7 (246/252) | 66.7 (213/213) |
| 2, greedy, 512 | 23.2 | 63.3 (1126/1187) | 68.9 (1208/1284) | 74.6 (1257/1359) | 80.3 (1293/1419) | 84.6 (1320/1467) | 63.8 (1134/1200) |
| 2, seeded, 96 | 23.2 | 59.5 (204/234) | 61.3 (216/264) | 65.0 (225/282) | 68.9 (231/297) | 69.1 (234/318) | 60.1 (204/234) |
| 2, seeded, 512 | 23.1 | 54.1 (1056/1383) | 55.6 (1131/1576) | 56.2 (1158/1784) | 55.8 (1183/1954) | 57.9 (1199/2075) | 55.1 (1073/1441) |

group of two, default: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 32.0 | 46.8 (155/260) | 46.8 (121/219) | 32.2 (62/133) | 38.8 (104/171) | 45.2 (151/256) | 41.0 (105/183) |
| 1, greedy, 512 | 33.0 | 72.2 (1088/1262) | 62.4 (1040/1207) | 62.5 (1046/1205) | 60.9 (1024/1232) | 60.8 (1061/1299) | 61.6 (921/1050) |
| 1, seeded, 96 | 33.3 | 58.5 (183/297) | 51.0 (168/282) | 50.1 (173/326) | 50.5 (180/355) | 51.3 (183/350) | 51.4 (163/252) |
| 1, seeded, 512 | 32.9 | 66.4 (1056/1381) | 63.7 (1021/1255) | 60.3 (997/1253) | 59.1 (968/1228) | 62.2 (1129/1667) | 61.7 (1013/1297) |
| 2, greedy, 96 | 33.2 | 80.2 (211/211) | 81.3 (225/231) | 89.3 (237/240) | 94.5 (243/243) | 97.8 (246/252) | 81.0 (225/231) |
| 2, greedy, 512 | 33.0 | 74.5 (1104/1236) | 74.6 (1186/1353) | 77.1 (1230/1476) | 79.7 (1263/1569) | 81.0 (1287/1653) | 73.9 (1188/1361) |
| 2, seeded, 96 | 33.0 | 74.4 (204/234) | 71.2 (216/264) | 75.2 (225/282) | 77.4 (231/297) | 76.2 (234/318) | 70.0 (216/264) |
| 2, seeded, 512 | 32.9 | 60.6 (997/1458) | 59.4 (1056/1599) | 57.2 (1056/1740) | 58.0 (1026/1742) | 56.5 (1067/1905) | 58.8 (1027/1559) |

group of two, --dtype int8: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 32.0 | 41.2 (72/109) | 48.2 (175/425) | 40.1 (105/199) | 34.9 (60/112) | 34.1 (49/79) | 32.6 (34/63) |
| 1, greedy, 512 | 32.8 | 78.0 (1068/1182) | 80.6 (1114/1344) | 78.2 (1116/1292) | 66.5 (946/1116) | 65.7 (959/1153) | 66.5 (1042/1208) |
| 1, seeded, 96 | 32.9 | 51.5 (159/312) | 50.0 (157/319) | 50.5 (161/331) | 48.6 (161/359) | 49.7 (160/353) | 48.9 (158/359) |
| 1, seeded, 512 | 32.6 | 56.6 (917/1538) | 54.6 (883/1453) | 53.6 (804/1290) | 54.6 (880/1425) | 54.0 (865/1409) | 52.8 (792/1281) |
| 2, greedy, 96 | 32.8 | 85.4 (212/212) | 91.5 (225/231) | 86.1 (213/213) | 91.2 (225/231) | 112.9 (246/252) | 96.1 (233/237) |
| 2, greedy, 512 | 32.8 | 82.6 (1124/1184) | 89.6 (1208/1284) | 86.6 (1188/1275) | 90.7 (1218/1310) | 107.1 (1320/1467) | 97.2 (1257/1359) |
| 2, seeded, 96 | 33.0 | 77.9 (204/234) | 79.8 (216/264) | 85.7 (225/282) | 88.9 (231/297) | 88.6 (234/318) | 85.6 (225/282) |
| 2, seeded, 512 | 32.6 | 76.1 (1089/1286) | 79.4 (1165/1447) | 79.9 (1192/1548) | 86.1 (1244/1709) | 81.4 (1239/1781) | 80.5 (1192/1548) |

group of four, default: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 26.9 | 47.0 (167/342) | 48.0 (178/415) | 43.8 (180/497) | 37.4 (184/572) | 35.6 (102/166) | 46.6 (148/264) |
| 1, greedy, 512 | 17.9 | 68.8 (1090/1282) | 74.9 (1131/1400) | 62.9 (1170/1550) | 59.4 (1164/1640) | 70.8 (1054/1323) | 62.8 (987/1136) |
| 1, seeded, 96 | 18.1 | 50.1 (183/297) | 55.0 (186/350) | 34.9 (193/392) | 35.2 (185/351) | 56.1 (190/360) | 52.5 (165/247) |
| 1, seeded, 512 | 20.6 | 66.7 (1054/1386) | 70.7 (1119/1618) | 48.4 (1160/1711) | 55.7 (1157/1734) | 70.2 (1096/1511) | 66.5 (1050/1376) |
| 2, greedy, 96 | 17.8 | 77.9 (211/211) | 90.1 (225/231) | 64.3 (236/239) | 104.1 (243/243) | 105.6 (243/243) | 90.0 (225/231) |
| 2, greedy, 512 | 20.9 | 75.0 (1104/1236) | 84.7 (1185/1353) | 57.1 (1229/1476) | 86.8 (1263/1569) | 87.6 (1263/1569) | 83.2 (1189/1361) |
| 2, seeded, 96 | 31.8 | 73.6 (204/234) | 79.9 (216/264) | 60.4 (225/282) | 83.5 (231/297) | 84.1 (231/297) | 79.5 (216/264) |
| 2, seeded, 512 | 28.9 | 60.5 (997/1458) | 65.7 (1088/1749) | 42.9 (1122/2026) | 59.8 (1122/2127) | 61.9 (1100/1985) | 62.1 (1076/1753) |

group of four, --dtype int8: tok/s (drafts kept/fed)

| prompt, sampler, tokens | off | depth 3 | depth 4 | depth 5 | depth 6 | depth 7 | depth 8 |
|---|---|---|---|---|---|---|---|
| 1, greedy, 96 | 29.5 | 27.9 (72/140) | 31.8 (81/143) | 38.0 (106/207) | 26.8 (5/24) | 28.3 (69/143) | 26.4 (5/26) |
| 1, greedy, 512 | 30.5 | 69.1 (1067/1377) | 66.9 (1030/1267) | 58.2 (1037/1370) | 61.6 (1007/1259) | 63.7 (1028/1372) | 56.9 (963/1253) |
| 1, seeded, 96 | 31.2 | 38.1 (95/278) | 34.8 (116/412) | 26.5 (60/337) | 35.1 (123/471) | 35.8 (120/442) | 36.1 (123/424) |
| 1, seeded, 512 | 31.0 | 47.0 (836/1725) | 46.2 (815/1694) | 41.4 (723/1462) | 44.1 (785/1561) | 42.0 (812/1790) | 44.7 (812/1649) |
| 2, greedy, 96 | 30.2 | 78.8 (212/212) | 89.6 (225/231) | 102.6 (237/240) | 106.1 (243/243) | 109.3 (244/250) | 107.3 (249/264) |
| 2, greedy, 512 | 30.7 | 79.3 (1125/1177) | 87.3 (1203/1295) | 98.4 (1254/1374) | 106.1 (1293/1419) | 110.8 (1323/1446) | 106.2 (1344/1476) |
| 2, seeded, 96 | 30.3 | 76.9 (204/234) | 78.9 (216/264) | 86.6 (225/282) | 90.2 (231/297) | 88.9 (234/318) | 84.4 (237/330) |
| 2, seeded, 512 | 30.5 | 62.5 (1018/1491) | 65.6 (1091/1733) | 63.3 (1126/2004) | 64.6 (1122/2131) | 62.4 (1147/2431) | 59.5 (1093/2046) |

Affected figures kept as run (group of four, `--dtype int8`, tok/s with kept/fed):

| prompt, sampler, tokens | depth 3 (affected) | depth 4 (affected) |
|---|---|---|
| 1, greedy, 96 | 37.8 (94/178) | 35.2 (70/117) |
| 1, greedy, 512 | 67.7 (1066/1377) | 66.9 (990/1285) |
| 1, seeded, 96 | 36.2 (95/295) | 35.5 (116/373) |
| 1, seeded, 512 | 44.8 (789/1518) | 44.7 (812/1661) |
| 2, greedy, 96 | 76.0 (213/213) | 92.3 (225/231) |
| 2, greedy, 512 | 78.4 (1128/1171) | 86.5 (1205/1296) |
| 2, seeded, 96 | 73.1 (204/234) | 79.3 (216/264) |
| 2, seeded, 512 | 58.5 (1020/1488) | 63.5 (1093/1721) |

Where each depth won, over the eight cells of a shape and precision (the relative figure is the mean of the eight ratios to depth 3):

| shape, precision | cells won by depth | tok/s relative to depth 3, depths 3 / 4 / 5 / 6 / 7 / 8 |
|---|---|---|
| one MI50, default | {3: 5, 4: 1, 6: 1, 7: 1} | 3 1.00 4 0.94 5 0.93 6 0.92 7 0.92 8 0.87 |
| one MI50, --dtype int8 | {3: 2, 4: 1, 6: 3, 7: 2} | 3 1.00 4 1.02 5 1.00 6 1.09 7 1.09 8 0.99 |
| layer split of two, default | {3: 5, 6: 3} | 3 1.00 4 0.94 5 0.92 6 0.94 7 0.95 8 0.87 |
| layer split of two, --dtype int8 | {3: 2, 4: 1, 7: 4, 8: 1} | 3 1.00 4 1.02 5 1.05 6 1.08 7 1.10 8 0.98 |
| group of two, default | {3: 5, 6: 1, 7: 2} | 3 1.00 4 0.96 5 0.93 6 0.96 7 0.99 8 0.93 |
| group of two, --dtype int8 | {3: 2, 4: 2, 6: 2, 7: 2} | 3 1.00 4 1.05 5 1.01 6 1.01 7 1.05 8 1.00 |
| group of four, default | {4: 4, 7: 4} | 3 1.00 4 1.09 5 0.80 6 0.98 7 1.08 8 1.04 |
| group of four, --dtype int8 | {3: 3, 4: 1, 5: 1, 6: 1, 7: 2} | 3 1.00 4 1.04 5 1.06 6 1.08 7 1.09 8 1.05 |

Drafts kept/fed by draft position (position 1 first) for the 512-token greedy cells, depth 3 and depth 8, by shape and precision. The depth fed falls with the position because the server prices a draft by its pass cost.

| shape, precision, prompt | depth 3 | depth 8 |
|---|---|---|
| one MI50, default, prompt 1 | 1:420/458 2:391/458 3:255/344 | 1:465/506 2:413/458 3:127/186 4:1/4 5:1/4 6:1/4 7:1/4 8:1/4 |
| one MI50, default, prompt 2 | 1:387/415 2:368/412 3:349/409 | 1:496/529 2:481/526 3:5/6 4:5/6 5:4/6 6:4/6 7:3/6 8:3/6 |
| one MI50, --dtype int8, prompt 1 | 1:391/429 2:362/429 3:336/427 | 1:465/507 2:332/384 3:42/82 4:37/66 5:35/60 6:33/55 7:32/54 8:32/54 |
| one MI50, --dtype int8, prompt 2 | 1:386/397 2:375/396 3:364/396 | 1:180/189 2:174/189 3:168/189 4:168/186 5:165/186 6:165/186 7:165/186 8:156/186 |
| layer split of two, default, prompt 1 | 1:390/428 2:362/428 3:338/426 | 1:466/507 2:412/459 3:121/164 4:2/5 5:2/5 6:2/5 7:2/5 8:2/5 |
| layer split of two, default, prompt 2 | 1:387/415 2:368/412 3:349/409 | 1:494/527 2:479/524 3:5/6 4:5/6 5:5/6 6:5/6 7:5/6 8:5/6 |
| layer split of two, --dtype int8, prompt 1 | 1:391/429 2:362/429 3:336/427 | 1:501/540 2:465/540 3:14/47 4:2/6 5:2/6 6:2/6 7:2/6 8:2/6 |
| layer split of two, --dtype int8, prompt 2 | 1:385/396 2:375/396 3:366/395 | 1:387/396 2:373/396 3:364/393 4:2/3 5:2/3 6:2/3 7:2/3 8:2/3 |
| group of two, default, prompt 1 | 1:398/435 2:369/435 3:321/392 | 1:553/596 2:319/369 3:20/35 4:13/22 5:4/7 6:4/7 7:4/7 8:4/7 |
| group of two, default, prompt 2 | 1:387/415 2:368/412 3:349/409 | 1:320/342 2:305/339 3:290/336 4:269/336 5:1/2 6:1/2 7:1/2 8:1/2 |
| group of two, --dtype int8, prompt 1 | 1:413/445 2:332/369 3:323/368 | 1:437/474 2:294/336 3:136/172 4:129/155 5:43/68 6:1/1 7:1/1 8:1/1 |
| group of two, --dtype int8, prompt 2 | 1:384/395 2:375/395 3:365/394 | 1:264/273 2:261/273 3:252/273 4:243/270 5:237/270 6:0/0 7:0/0 8:0/0 |
| group of four, default, prompt 1 | 1:390/428 2:362/428 3:338/426 | 1:501/546 2:459/522 3:14/40 4:3/6 5:3/6 6:3/6 7:2/5 8:2/5 |
| group of four, default, prompt 2 | 1:387/415 2:368/412 3:349/409 | 1:320/341 2:305/338 3:290/335 4:270/335 5:1/3 6:1/3 7:1/3 8:1/3 |
| group of four, --dtype int8, prompt 1 | 1:391/460 2:359/459 3:317/458 | 1:450/526 2:411/524 3:21/46 4:20/42 5:19/41 6:17/33 7:13/21 8:12/20 |
| group of four, --dtype int8, prompt 2 | 1:384/394 2:377/392 3:364/391 | 1:180/186 2:177/186 3:171/186 4:168/186 5:165/183 6:165/183 7:162/183 8:156/183 |

### Set 2: a round's rollback

Qwen3.8-27B Q8_0, default precision, `bench --model FILE --drafter embedded --p 512 --n 128 --r 10`: after a mark and a verify of 3 drafts, the retract keeping each of the verify's 4 rows and the step after it as completed work up to the step's logits; the difference to keeping all 4 rows in the same repeat, median of ten runs (the middle half in brackets), and the retract call's own host time. This is the measurement of the earlier record in this file (Rollback, the same file on one MI50: +1.09 to +1.14 ms on a step of about 44 ms). The three shapes ran one after another (15:24 to 15:32 UTC, load average 10 at the start).

| shape | step keeping all 4 rows, ms | keeping 1 of 4 | keeping 2 of 4 | keeping 3 of 4 |
|---|---|---|---|---|
| one MI50 | 42.636 | +0.822 ms, +1.93% (+0.713 to +0.869), call 0.546 ms | +0.773 ms, +1.81% (+0.696 to +0.922), call 0.512 ms | +0.912 ms, +2.14% (+0.827 to +0.976), call 0.613 ms |
| layer split of two | 42.910 | +0.471 ms, +1.10% (+0.121 to +0.923), call 0.432 ms | +0.369 ms, +0.86% (+0.027 to +0.804), call 0.413 ms | +0.501 ms, +1.17% (+0.214 to +0.705), call 0.453 ms |
| group of two | 30.057 | +0.712 ms, +2.37% (+0.445 to +1.271), call 0.801 ms | +1.183 ms, +3.93% (+0.507 to +1.509), call 0.792 ms | +0.640 ms, +2.13% (-0.231 to +1.288), call 0.782 ms |

### Set 3: the tensor split's open cells again, with a thread per member

Qwen3-32B Q8_0, a group of four (MI50s under one root) and a group of two (under another), the three arms in turn each round with the order rotated, two rounds. Each arm ran a bench (llmx `bench --p 512 --n 128 --r 3`; the reference `llama-bench -ngl 99 -fa on -sm tensor -lm dio -p 512 -n 128 -r 3`) and then a fresh server loaded at 1 and 4 users with 128-token prompts and 128 generated tokens (`tools/server_load.py`, one round, greedy; llmx `--max-seqs 8 --ctx-size 16384`, the reference `-c 16384 -np 8 -cram 0`). The reference is the pinned image's `llama-bench` and `llama-server` with the image's environment and 8 hardware queues; its log shows "creating a Meta device for tensor parallelism from 4 devices (tps=4, n_stages=1)" and from 2 devices (tps=2, n_stages=1), with its custom all-reduce initialised. Each cell gives the first round, then the second. No request failed. Load average 9.8 to 48.5 for the group of four (16:05 to 16:24 UTC) and 10.2 to 24.3 for the group of two (16:31 to 16:40 UTC), other sessions on other cores.

Group of four:

| arm | pp512 tok/s (r1, r2) | tg128 tok/s (r1, r2) | 1 user tok/s (r1, r2) | 1 user itl p99 ms | 4 users tok/s (r1, r2) | 4 users itl p99 ms | failed |
|---|---|---|---|---|---|---|---|
| llmx --dtype int8 | 589.11, 588.65 | 29.66, 32.38 | 31.0, 30.2 | 35, 36 | 98.7, 96.4 | 36, 41 | 0 |
| llmx default | 460.46, 461.17 | 33.08, 30.86 | 30.2, 28.8 | 37, 37 | 91.0, 90.9 | 39, 47 | 0 |
| reference | 522.32, 523.37 | 51.03, 50.75 | 43.5, 43.7 | 44, 43 | 105.3, 105.5 | 64, 60 | 0 |

Group of two:

| arm | pp512 tok/s (r1, r2) | tg128 tok/s (r1, r2) | 1 user tok/s (r1, r2) | 1 user itl p99 ms | 4 users tok/s (r1, r2) | 4 users itl p99 ms | failed |
|---|---|---|---|---|---|---|---|
| llmx --dtype int8 | 544.37, 546.93 | 29.83, 29.98 | 27.8, 28.4 | 37, 34 | 84.7, 85.6 | 40, 41 | 0 |
| llmx default | 363.61, 366.42 | 29.62, 29.97 | 26.9, 27.3 | 37, 36 | 74.0, 74.7 | 45, 44 | 0 |
| reference | 575.84, 575.70 | 33.81, 32.44 | 30.5, 30.6 | 52, 47 | 79.7, 80.0 | 68, 68 | 0 |

### Set 4: the layer split of three, the loaded cells rerun

Qwen3-8B Q8_0, three MI50s, llmx `--dtype int8` and default precision (host tier off, as in the earlier gate) against the reference's `-sm layer`, one session (16:46 to 17:16 UTC), arms in rotation, `tools/server_load.py` closed loads of two rounds at 1 to 64 users with 128 generated tokens on 81920 tokens over 64 sequences; skew: 16 users of 128-token prompts beside four 4096-token prompts, 131072 tokens over 20 sequences; near the context: four 16000-token prompts on 65536 tokens over 4 sequences. One of the earlier three-card tables' cards was inside another session's window now, so this session used another card in its place (still one card under a different root than the other two). The greedy 128-token set ran for all three arms. Load average 6.3 to 17.2.

Closed loads, output tok/s (lower and higher round), with the first-token and inter-token tails:


| 128-token prompts, output tok/s (lower and higher round) | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| llmx `--dtype int8` | 68.3-68.6 | 133.6-134.1 | 227.0-229.0 | 391.9-392.5 | 593.7-617.4 | 789.1-809.0 | 934.7-934.9 | 855.3-865.9 |
| llmx, default precision | 64.2-64.3 | 125.5-126.4 | 196.7-198.0 | 319.8-320.1 | 450.6-456.0 | 497.4-505.9 | 584.7-584.8 | 534.9-539.5 |
| reference, layer split | 61.9-62.1 | 98.1 | 165.4-165.9 | 214.0-215.4 | 230.4-230.5 | 293.1-355.1 | 283.7-360.6 | 408.0-411.0 |

| 128-token prompts, at 16 / 32 / 48 / 64 users | time to first token p99, s | inter-token p50, ms | inter-token p99, ms |
|---|---|---|---|
| llmx `--dtype int8` | 0.89-1.02 / 1.49-1.62 / 2.11 / 2.76-2.89 | 19 / 28 / 35 / 53 | 23-24 / 141 / 423 / 471-489 |
| llmx, default precision | 1.30-1.35 / 2.17-2.32 / 3.10 / 4.06-4.19 | 25 / 47 / 59 / 89 | 28-29 / 213-214 / 637 / 692-715 |
| reference, layer split | 1.45-1.46 / 2.82-2.83 / 4.56-4.93 / 5.84-5.89 | 58 / 69-88 / 98-132 / 110-111 | 62 / 73-92 / 134-152 / 1360 |

| 128-token prompts: llmx's lower round over the reference's higher round at 16 / 32 / 48 / 64 users (p99s: T time to first token, I inter-token, each llmx's higher round against the reference's lower) | reference, layer split |
|---|---|
| llmx `--dtype int8` | 2.58 / 2.22 (I worse) / 2.59 (I worse) / 2.08, needs 2.0 |
| llmx, default precision | 1.96 / 1.40 (I worse) / 1.62 (I worse) / 1.30, needs 2.0 |

| 1024-token prompts, output tok/s (lower and higher round) | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| llmx `--dtype int8` | 48.0 | 85.2-85.3 | 126.9-132.1 | 194.7-196.7 | 243.9-245.7 | 278.8-281.4 | 298.3-299.3 | 288.1-289.1 |
| llmx, default precision | 40.0-40.2 | 71.5-71.6 | 102.2-102.8 | 148.4-150.4 | 174.6-175.7 | 187.6-188.6 | 199.8 | 193.2-193.3 |
| reference, layer split | 50.2-50.3 | 72.7-79.1 | 106.1-106.4 | 123.7-126.7 | 129.4-129.9 | 137.1-147.1 | 123.9-142.2 | 142.1-142.3 |

| 1024-token prompts, at 16 / 32 / 48 / 64 users | time to first token p99, s | inter-token p50, ms | inter-token p99, ms |
|---|---|---|---|
| llmx `--dtype int8` | 5.36-5.41 / 10.22-10.36 / 15.08-15.15 / 20.46-20.56 | 23 / 35 / 46 / 68 | 464-465 / 474-484 / 489-491 / 509-513 |
| llmx, default precision | 7.90-7.97 / 15.18-15.31 / 22.54 / 30.46-30.48 | 29 / 54 / 70 / 104 | 680 / 707-709 / 725-727 / 759-761 |
| reference, layer split | 7.93-7.96 / 18.19-18.30 / 29.99-30.88 / 41.32-41.66 | 62 / 76-96 / 109-161 / 124-127 | 1057-1061 / 1324-1350 / 1656-1707 / 1723 |

| 1024-token prompts: llmx's lower round over the reference's higher round at 16 / 32 / 48 / 64 users (p99s: T time to first token, I inter-token, each llmx's higher round against the reference's lower) | reference, layer split |
|---|---|
| llmx `--dtype int8` | 1.88 / 1.90 / 2.10 / 2.03, needs 2.0 |
| llmx, default precision | 1.34 (T worse) / 1.28 / 1.40 / 1.36, needs 2.0 |

No request failed in any closed load.

Skewed load (16 users beside four long prompts), users' rounds in order:

| arm | long prompts' first token, s | users' tok/s per round | users' inter-token p99 per round, ms |
|---|---|---|---|
| llmx --dtype int8 | 4.5, 4.9, 8.9, 4.7 | 587.3, 618.2, 163.7, 590.2 | 25, 20, 647, 25 |
| llmx default | 6.7, 12.4, 6.1, 6.4 | 446.5, 122.6, 409.8, 454.9 | 29, 871, 36, 29 |
| reference layer split | 5.3, 3.1, 8.9, 7.5 | 118.5, 203.3, 173.2, 169.3 | 1103, 86, 113, 86 |

Four 16000-token prompts near the context, two rounds:

| arm | output tok/s | first token p50, s | inter-token p99, ms | failed |
|---|---|---|---|---|
| llmx --dtype int8 | 8.5, 8.5 | 29.4, 28.9 | 1328, 1329 | 0 |
| llmx default | 6.9, 6.9 | 36.3, 36.1 | 1553, 1553 | 0 |
| reference layer split | 11.0, 11.0 | 26.2, 25.8 | 1769, 1769 | 0 |

### Not run

- Two stages of four on eight cards (the request was the groups of four and two).
- Set 4 on the exact cards of the earlier three-card tables (one was in another session's window).

## The run list a routed layer rebuilds is each device's own (2026-10-08, branch fix/run-list-per-device, lands by fast-forward)

- **Fault:** a routed layer rebuilds the run list of its rows for the SiLU between its projections (`blocks::routed_experts`) in a list the whole context held (`ExecContext::entry_runs`), and two stages of tensor groups record on a thread each at two or more passes in flight, so two stages recording routed layers at once cleared and filled one list. A stage could then read another pass's runs. Reachable since the tensor split runs routed experts (two or more stages of groups, a routed file, `--passes` above 1); one stage, which production runs, is not exposed, and a layer split has no stage threads on main.
- **Why no test met it:** no hosted test served a routed model over two stages of groups with passes in flight.
- **Failing test first:** `server-passes-cpu` serves the synthetic model as a routed file (`served_routed`: 4 experts a layer, 2 a token) over two stages of CPU groups at P = 2 and 4, every reply its reply alone on one group. Under the thread sanitizer on the test machine (g++ 14, `-fsanitize=thread -g`, address randomization off) the test commit exits 66 with 6 reports, each a data race on the list under `routed_experts` from `Model::group_record`; at the fix it exits 0 with none.
- **Fix:** the list is each device's, beside its arena (`ExecContext::Scratch::entry_runs`), since a device is recorded on by one thread at a time.
- **Gates at the fix, fresh builds on the test machine:** the Vulkan build without a warning, CTest 46 of 46 with one MI50, the CPU suite, `docs` and `dead-code`. The many-user check's ids and log-probabilities on Qwen3-8B Q8_0 over two MI50s are main's, 502992 bytes, measured at the branch's earlier head, whose model code is this one's. No kernel and no row's arithmetic changes.
- **Found beside it and not fixed here:** the embedded drafter's chain is work on the head's group that the scheduler's thread does as it forms a pass, with no wait for that stage's thread. On a model that keeps a state, which an embedded drafter needs, a chain that follows a verify is behind that verify's retract, which waits; a chain after a pass that was no verify, the first draft after a prompt among them, is not. Two or more stages of groups with `--drafter embedded` at `--passes` above 1 are exposed; one stage is not. A latch case parking forty recordings on the head's group met no call beside them, so there is no failing case yet and no fix on main.
- **Review:** F2DEV.

## Disk files lost or damaged between two servers, the segment and state counters, and the long read profiled (2026-10-08, branch feat/disk-increment-4, steps 4 and 5 of DISK-TIER's Entries written as what changed, lands by fast-forward)

- **Done, the fault cases:** `server-resume` starts a scheduler on a copy of a kept directory with one fault in it, on the dense model and on the hybrid one: a segment gone from the middle of the path, the first segment gone, a temporary file a crash left, a byte flipped in a segment's payload, and on the hybrid model the newest state gone and every state gone. Each start keeps exactly the files a whole path from an empty history reaches, and the next turn forks what they give with the reply of a fresh model. With disk room for a first turn's two segments and no more, the second turn's segment is not written and nothing is deleted for it. `disk-index` holds the age limit to an expired conversation's own files where a younger branch shares its base.
- **A defect the cases found, fixed here with its test first:** a read whose second file failed its checksum counted two errors, the files after it on the path, deleted with it, failing in turn as missing; a read now counts one error, at its first file that fails. The test commit fails on main at "a byte flipped in the second segment: the next turn reused 0 tokens with 2 errors".
- **Done, the counters and lines:** `/v1/health` gives `reuse.disk.now.segments` and `.states` beside `.entries`; when an idle server's writes end it prints what went to disk since they last ended and the files there; a stop that could not write everything for want of room says so.
- **The idle time is an option tests shorten** (`DiskOptions::idle`, `kDiskIdle` by default, no flag): the hosted Windows job stopped at its 25-minute limit on this branch's first form, main's own run of it taking 1426 and 1445 s of the 1500, so the new cases and `disk_increment` wait one idle second in place of five and `server-resume` is no longer than it was. The job's margin on main was about a minute, and its limit has since been raised (the record below).
- **Not built, decided here for the reviewer to accept or reject:** an order that takes a regenerated reply's old tail before other conversations' files. The index cannot tell such a tail from a conversation that shares a prefix with another, and the tail goes by its own last use as every leaf does.
- **The long read, profiled before any change** (the coordinator: the restart's next turn read 82 files at about 670 MB/s against 852 for one file). The 852 was production's read, on its own disk, so the two figures were never one environment's. Matched, on the test machine: the whole copy that was (main `761a38286`) against the path of files (`7a24c4560`), timing builds with timers in the store's read, two MI50s, Qwen3.8-27B Q8_0 with production's flags, a 76k-token turn written while idle, then three starts an arm, interleaved, each reading the conversation back for its next turn:

  | arm | files | waited for the read (ms) | first token (s) | slabs taken (ms) | store: open | header | read | checksum | copy (ms) |
  |---|---|---|---|---|---|---|---|---|---|
  | one file | 1, or 2 with a boundary's state | 7073, 6836, 6901 | 8.73, 8.44, 8.54 | 2249, 2302, 2084 | 0 | 3 to 7 | 3066 to 3240 | 718 to 755 | 741 to 824 |
  | the path | 76 | 6810, 6796, 6683 | 8.46, 8.46, 8.29 | 2091, 2069, 1900 | 7 | 161 to 163 | 3073 to 3082 | 666 to 729 | 756 to 791 |

  The path is not slower: 6.68 to 6.81 s waited against 6.84 to 7.07 s. Its files cost 0.16 s, the 76 headers of 1 MiB at 2 ms each, and opening them 7 ms; the runs a segment is cut into cost nothing. The store alone, outside the server, the same 5.2 GiB on the same pool: one file 4.34 s (1263 MB/s), 82 files 4.54 s (1209 MB/s), the 0.2 s again the headers.
- **Where a long read's 6.8 s goes, for both layouts:** 2.0 s taking 5.2 GiB of host-visible slabs, on the scheduler thread before the read starts; 3.1 s reading around the file cache, the pool's rate; 0.7 s of checksums and 0.8 s copying into the slabs, one after the other on the store's one thread. Two candidates follow from it, neither built: slabs taken off the scheduler thread or kept from the stop of the write before, and the checksum and copy of a chunk overlapping the next chunk's read, worth at most the 1.5 s they take.
- **Measured, gates:** before the rebase onto `a0922bb62`, at the same src and tests: Qwen3-0.6B Q8_0 gives main's ids and logits on the CPU and on one MI50, and the suite's `server` component and `server-resume` with its device half pass on the MI50. Before the idle option, at the same src otherwise: on the test machine a CPU build, CTest, the whole CPU suite and the linked check. At the landing head: the hosted run, and on Windows CTest with `docs`, `dead-code`, `version` and `arch-boundary`.
- **Reviewed:** by F2DEV at the test commit and the change before the rebase: no finding in the code; the four decisions put to it accepted (the states above a lost segment staying until the next start, the faulted request reusing nothing, no order for a regenerated reply's old tail, the idle line once a period); of the read's two candidates it names the slabs taken on the scheduler thread worth a branch, a serving defect, and the overlap a record.

## The short-donor cancel case cancels from the scheduler's thread (2026-10-09, branch test/short-donor-cancel, tests only, lands by fast-forward)

- **Why:** `server-resume` failed twice in a row in the hosted macOS job on a commit that changes only tests/qwen35.py (runs 37856712190 and 37862276521), "a paused request with a short donor cancelled: 1 pauses, 1 taken back". The case's thread spun on the pause count and then cancelled the paused request; a thread scheduled late found the request already resumed.
- **Done:** the cancel is made in `on_retire`, on the scheduler's thread, as the first pass after the pause retires, so no round can admit the paused request between its pause and its cancel. The case's assertions are unchanged.
- **Shown, in a Windows build:** with the test's thread held 200 ms before it acts, the old form fails ("ended with length") and the new form passes; unheld, the CPU half passes three runs of three.
- **Left:** the other timing-shaped cases of the review's audit of 2026-10-09, which get one branch of their own after feat/pool-edge and the kept-conversations work; this one went ahead because it was failing hosted runs.
- **Reviewed:** by O5REV before landing.

## The hosted TSan job's limit is 25 minutes (2026-10-09, branch ci/tsan-limit, the workflow and its page only, lands by fast-forward)

- **Why:** the job's thread tests took 591 and 744 s on main `8278c6b7f` and `5472fcf91`, with a minute of build before them, against a limit of 900 s for the job, and were ended at the limit after 818 s on perf/draft-while-waiting, which adds no work to `http` or `server-passes-cpu`; every other job of that run passed.
- **Done:** `timeout-minutes` for the TSan job is 25, and `docs/CI.md` gives the runs. No test, check or source changes.
- **Left:** the job's time itself, which varies by a quarter between two runs of main.
- **Reviewed:** by F2DEV before landing.

## The resume-drafts check sends its second request sooner (2026-10-09, branch test/resume-drafts-margin, tests only, lands by fast-forward)

- **Why:** the suite's `qwen35` component failed once in a hosted Windows job on a workflow-only commit (run 37849561804), in `check_resume_drafts`: the second request was paused inside a block, at 1330 tokens, and recomputed, so no copy was promoted from host memory, three tries of three. The check needs the second request at its reservation's end, 384 tokens on, before the first one's growth step, 511 on, and it sent the second only when the client had read the first one's first token, so the stream's delivery came out of 127 passes of a model whose pass is under a millisecond.
- **Done:** the second request is sent once `/v1/health` shows the first admitted, while it still reads its 5762-token prompt. What the check asserts is unchanged.
- **Measured, on Windows, a delay added before the second request is sent, one run each:** the old form passes at 0, 0.03, 0.1 and 0.15 s and fails at 0.2 s and above; the new form passes through 0.5 s and fails at 1 s. Under 32 busy loops on 16 hardware threads both forms passed, so plain CPU load is not what the runner met.
- **Tried and not taken:** the prompt read four tokens a pass passes at 1 s and fails at 2 s, for twice the check's time; one token a pass fails at every delay, the second request being paused before the first decodes.
- **Left:** it is still a check by timing, with half a second where it had a seventh; the path itself is held without a clock by `server-spec`'s resumed-by-fork case.

## The hosted Windows job's limit is 35 minutes (2026-10-08, branch ci/windows-limit, the workflow and its page only, lands by fast-forward)

- **Why:** the job took 1426 and 1445 s of its 1500 on main `a0922bb62` and `07c21c6ee`, and was ended at the limit twice on feat/disk-increment-4 with every step passing, once after its last step; that branch's native tests take what main's do there, 550 s against 549 and 550.
- **Done:** `timeout-minutes` for `windows-2022` is 35, and `docs/CI.md` gives the runs. No test, check or source changes.
- **Left:** the job's time itself, 6 minutes of build, 9 of native tests and 6 and a half of the suite.
- **Reviewed:** by F2DEV before landing.

## The tensor split runs routed experts (2026-10-07, branch feat/tp-moe, the tensor split's step 8, lands by fast-forward)

- **Goal:** `--tensor-width N` on the mixture-of-experts models the layer split serves, qwen3moe and qwen35moe, which a group refused (`docs/TENSOR-SPLIT.md`, step 8).
- **Done:** a routed layer's expert stacks declare their split, each expert's hidden rows (`blocks::shard_experts`: the gate and up stacks by the rows of each expert, the down stack by as many columns), qwen35moe's shared expert as a dense block, the router and the shared expert's gate vector whole; `shard::extent` and `shard::share` read an expert stack's rows over its experts; a member's routed sum goes into its partial rows, which the runtime clears first from zero rows the context keeps (`Model::clear_partial`), and the shared expert's down projection adds to them (`blocks::join` with `add`), so a part ends in one sum; `check_plan` no longer refuses a routed layer and refuses an expert width off whole blocks of its down stack as it refuses any column split. An embedded drafter whose block is routed clears its partial rows the same way. Nothing changed in a backend, the collective, the scheduler or the server; width 1 keeps the order of its two residual adds.
- **Which form and why, and which files split:** `docs/TENSOR-SPLIT.md`, Step 8 as built. Qwen3.6-35B-A3B splits at width 2 in a K-quant and at 2 and 4 in Q8_0; Qwen3-30B-A3B at 2 and 4 in Q8_0 and at no width in a K-quant, where it is refused naming `ffn_down_exps`.
- **Correctness on CPU groups** (hosted): two new fixtures with HF goldens (`docs/ASSETS.md`, Tensor-split fixtures), a qwen3moe and a qwen35moe whose every split falls whole at widths 2 and 4: against HF at both widths the five tensor-split fixtures stay within 4.5e-7 logits (bound 2e-5) and 4e-8 NLL (bound 1e-5), and each passes the device-reference criterion against one backend; `llmx-split-check` on both as two stages of groups of two against one group, bit-identical, slices, decode steps, passes in flight and verifies included; the embedded drafter of the qwen35moe fixture with an MTP block, its picks and drafts those of one backend; `model-validation`: a routed plan taken at widths 2 and 4 and an expert width off whole Q8_0 blocks refused by its text.
- **Correctness on cards** (MI50s under one root, RADV, default precision, the branch before its last rebase with the loader change, every command's first line naming its group):

  | check | result |
  |---|---|
  | the five tensor-split fixtures on a group of two and of four, against HF and against one card | max logit error 5.2e-7 and 5.6e-7 (bound 2e-5), NLL error 8e-8 and 7e-8 (bound 1e-5); the routed drafter fixture's picks and drafts one card's |
  | Qwen3.6-35B-A3B Q4_K_M at width 2 against its file-exact and its layered goldens (`tests/baseline_layered.py`) | 63 of 63 |
  | Qwen3.6-35B-A3B Q8_0, 96 greedy and 96 seeded tokens on a group of two, a group of four and a layer split of two | the same ids on all three |
  | the many-user mix on Qwen3.6-35B-A3B Q4_K_M at width 2 with log-probabilities (`tools/server_mix_check.py`) | 0 of 8 differ together, 0 of 6 skewed, the CLI's text |
  | Qwen3-30B-A3B Q4_K_M at width 2, Qwen3.6-35B-A3B Q4_K_M at width 4 | refused |

  The witness that the members route alike on a card is the pair of rows 1 and 2 above: the tensor-split fixtures on a group (`run_tests.py --only tensor-split --device vulkan:0,vulkan:1 --tensor-width 2`, and 4 on four cards) held to HF and to one card by the device-reference criterion, and Qwen3.6-35B-A3B Q4_K_M against its layered goldens, since members choosing other experts would give logits that are neither HF's nor one card's.
  A planted skew of one member's router is not run.
  The 16k check was not run for this family: its prompt and decode kernels are the dense models', and what is new, the routed sum into partial rows, is held by the checks above.
- **Width 1 against main** (6f71114ca, both built the same way): ids and logits the same on one MI50 for Qwen3-30B-A3B Q4_K_M and Qwen3.6-35B-A3B Q4_K_M; Qwen3-30B-A3B Q4_K_M, main, branch, branch, main twice: pp512 1124.5 to 1126.6 against 1125.1 to 1126.7 and tg128 125.7 to 126.6 against 125.5 to 126.4 tok/s.
- **Speed, recorded at first support** (Qwen3.6-35B-A3B, MI50s under one root, default clocks, four whole cores, `bench --model --p 512 --n 128 --r 3`, two rounds, tok/s; the reference its ROCm tensor split of two on the same cards in the same session, its log showing the group formed; measured before the member threads landed):

  | Qwen3.6-35B-A3B Q4_K_M, two MI50s | pp512 | tg128 |
  |---|---:|---:|
  | llmx group of two, `--dtype int8` (the matched precision) | 2157.9, 2155.7 | 81.1, 79.9 |
  | llmx group of two, default precision | 1660.4, 1659.1 | 81.5, 81.4 |
  | reference tensor split of two | 1648.5, 1664.7 | 89.3, 95.8 |
  | llmx layer split of two, `int8` | 1552.0, 1552.5 | 111.9, 111.0 |
  | llmx one card, `int8` | 1596.0, 1604.6 | 116.7, 116.9 |

  One round, Qwen3.6-35B-A3B Q8_0 on four cards at `int8`: a group of four 1977.3 and 30.9, two stages of two 2156.8 and 79.3, a layer split of four 1653.6 and 98.8.
  So a group reads a mixture-of-experts prompt 1.35 times as fast as one card and 1.3 times the reference, and decodes below one card and below a layer split, as the plan expected of a model whose decode is bound by dispatches a group does not divide (`docs/TENSOR-SPLIT.md`, section 4.10 as it was): the tensor split serves this family's prompts, and the layer split its decode.
- **Open against the reference at the same precision:** decode on a group of two, 80 to 81 against 89 to 96 tok/s; recovery as for the dense models, a decode sum's cost.
- **The files this step refused**, Qwen3-30B-A3B in a K-quant at every width and Qwen3.6-35B-A3B in a K-quant at width 4, run since the covering blocks' branch (its record above).
- **Found on cards and fixed here, a refusal's text:** Qwen3-30B-A3B Q4_K_M at width 2 was refused with the quantization's own error, "a row of 384 values is not whole Q6_K blocks", since `place_model` read a member's shards for the fit before the model checked the width; `place_model` now checks the width first (`shard::check_plan`), so the refusal names the projection, with its failing test as the branch's first commit (a dense Q8_0 model whose feed-forward width leaves a member 48 columns, which main refused the same wrong way).
- **Found on cards and fixed here:** a group's load of Qwen3.6-35B-A3B Q4_K_M took three minutes, the streamed loader issuing a device copy for each row of each expert's column shard, about half a million of 210 bytes a down stack a member; runs shorter than a page are now packed on the host and written once a part (`detail::stream`), and the load takes 17 s with the first reply. A dense model's row and column shards, whose runs are kilobytes, load as they did. A group's streamed load runs in the hosted tests on CPU groups, through `llmx-model-logits` in the tensor-split component; the device runs of its fixtures and of the real files, held to HF, cover the copying backend.
- **Reviewed by:** F2DEV at dc78b2d5d, the same routed code: two should-fixes taken, the refused files as open cells with their recovery in the status table and the rule in USAGE, and the checks named that witness the members routing alike; the loader's packed writes and the width checked before the fit were added after that review and are in the delta re-requested.
- **Gotchas:** a member of a group that runs routed experts keeps one more residual row a row of a pass, the zero rows, which the fit counts.

## The embedded drafter runs over a tensor group (2026-10-07, branch feat/tp-spec, the tensor split's step 6, lands by fast-forward)

- **Goal:** `--drafter embedded` under a tensor width, which a group refused (`docs/TENSOR-SPLIT.md`, step 6); drafting by lookup came with step 5.
- **Done:** `check_plan` checks the drafter's block as a layer and no longer refuses it; every member of the head's group holds its shards of the block, the block's input projection and norms whole and the head whole (`member_drafter_`), its own carried rows and its marks' normed rows; `draft_context` writes each member's K and V of its own KV heads after the last stage; `draft` runs a step in three calls a member (`DraftStep`, `Step::phase`) with the group's sum after the block's mixer and after its feed-forward block, into the step's residual rows (`ModelPlan::draft_x`), and ends a sequence's drafts before an id its members do not agree on; the fit counts a member's shards of the block, the head once more and the KV of its KV heads, and counts a tied head on a group as the member's shard beside the whole embedding table, which it had counted as one buffer. `llmx-decode-probe` takes a tensor width.
- **The one departure from the plan:** the chain's argmax is not joined over the members; each member reads the head whole (`docs/TENSOR-SPLIT.md`, Step 6 as built, with the reason and the cost). No call was added to `Collective`, and nothing changed in a backend, the scheduler or the server.
- **Correctness on CPU groups** (hosted): `spec`: the embedded drafter of the hybrid model in the shape a group of two splits whole, on one and two stages of groups at 1, 3 and 8 drafts, greedy and seeded, from prompts of 10, 127 and 129 tokens, each run feeding drafts and giving the run without drafts of the file without the block on one group; a member's footprint with the drafter against a hand count; `tensor-split`: the qwen35 fixture with an MTP block on groups of two and four, from four prompts the pick and both drafts those of one backend and every draft row's logits within 2.2e-7 of one backend's (bound 2e-5), whose own drafts the `qwen35` component holds to HF; `shard`: a tied head on a group counted as two buffers.
- **On cards, the condition for this form** (Qwen3.8-27B Q8_0, MI50s under one root, default clocks, four whole cores with nothing else on them, 14025236e, `llmx 0.1.0+g14025236ec11`; `generate --file` of a 1500-byte prompt, 96 tokens, `--drafter off` against `embedded`, default precision, one run each; every run's first line names its group):

  | Qwen3.8-27B Q8_0, 96 tokens | drafts off, tok/s | embedded drafter, tok/s | gain | drafts kept at depth 1, 2, 3 |
  |---|---:|---:|---:|---|
  | one MI50, greedy | 23.0 | 35.9 | 1.56 | 26, 19, 12 of 38 |
  | group of two, greedy | 31.1 | 46.4 | 1.49 | 26, 19, 12 of 38 |
  | group of two, seed 7 at the defaults | 30.9 | 52.5 | 1.70 | 26, 21, 15 of 33 |
  | group of four, greedy | 20.3 | 36.2 | 1.78 | 26, 19, 12 of 38 |
  | group of four, seed 7 at the defaults | 20.0 | 40.7 | 2.04 | 26, 21, 15 of 33 |

  The ids with the drafter are the ids without it in every row, greedy and seeded, and the drafts kept on a group are those kept on one device, draft for draft. So drafts gain on a group of four as on one device with the head read whole, the condition under which this form lands without the join (F2DEV's review and the coordinator's word); a draft row's own time was not isolated, the gain being the measure asked.
  `bench --model --drafter embedded` at `--dtype int8`: pp512 and tg128 as without the drafter (637.6 and 32.8 against 640.8 and 33.3 on a group of two, 618.9 and 21.4 against 620.7 and 21.4 on a group of four), and a retract inside a mark 0.77 ms on a group of two and 1.7 ms on a group of four against 0.4 to 0.7 ms on one device, within 5 percent of a step.
- **Against the reference** (the same model and cards, one session, both sides as servers with one slot and a fresh server an arm, the same 1500-byte prompt, replies of 96 and 512 tokens with the end token ignored, greedy and seeded at temperature 0.8, top-k 40, top-p 0.95 and seed 7 with each side's own generator, three requests a cell and the median shown; each side's own decode rate and its own count of drafts kept of drafts fed over the three requests; llmx `serve --max-seqs 1 --ctx-size 8192` with `--drafter embedded` against off; the reference with its image environment and `--spec-type draft-mtp` against none at its default depth, `-sm tensor` for the groups and `-sm layer` for the split, its server log naming the tensor split mode and the MTP draft context; machine load 10 to 50 from other cores):

  | Qwen3.8-27B Q8_0, tok/s, drafts off / on (drafts kept of fed) | reference | llmx, `--dtype int8` (the matched precision) | llmx, default precision |
  |---|---|---|---|
  | one MI50, greedy, 96 tokens | 22.1 / 30.5 (171 of 327) | 23.3 / 38.2 (167 of 342) | 23.2 / 36.4 (164 of 334) |
  | one MI50, greedy, 512 | 22.1 / 40.0 (1071 of 1386) | 23.6 / 57.5 (1089 of 1285) | 23.4 / 52.9 (1088 of 1290) |
  | one MI50, seeded, 96 | 22.2 / 25.5 (150 of 399) | 23.6 / 33.8 (137 of 378) | 23.5 / 43.1 (183 of 297) |
  | one MI50, seeded, 512 | 22.1 / 32.1 (957 of 1722) | 23.0 / 35.5 (732 of 1453) | 23.4 / 49.7 (1058 of 1374) |
  | group of two, greedy, 96 | 36.0 / 52.7 (192 of 276) | 32.1 / 40.4 (72 of 109) | 32.8 / 47.9 (168 of 336) |
  | group of two, greedy, 512 | 36.5 / 67.0 (1122 of 1224) | 33.2 / 77.5 (1068 of 1182) | 33.2 / 72.3 (1092 of 1277) |
  | group of two, seeded, 96 | 36.0 / 40.8 (159 of 378) | 33.2 / 51.2 (159 of 315) | 33.2 / 57.9 (184 of 294) |
  | group of two, seeded, 512 | 36.4 / 47.2 (942 of 1767) | 33.1 / 56.3 (923 of 1564) | 33.4 / 66.0 (1053 of 1389) |
  | group of four, greedy, 96 | 49.1 / 58.8 (174 of 324) | 20.6 / 23.0 (78 of 136) | 20.4 / 36.8 (167 of 342) |
  | group of four, greedy, 512 | 50.6 / 83.1 (1092 of 1323) | 21.0 / 50.6 (955 of 1183) | 20.8 / 52.8 (1090 of 1282) |
  | group of four, seeded, 96 | 49.1 / 51.0 (150 of 405) | 21.2 / 27.3 (95 of 314) | 21.0 / 43.2 (183 of 297) |
  | group of four, seeded, 512 | 50.7 / 70.7 (1008 of 1572) | 21.2 / 33.8 (819 of 1631) | 20.8 / 49.3 (1054 of 1386) |
  | layer split of two, greedy, 96 | 21.7 / 30.0 (171 of 327) | 22.8 / 40.6 (167 of 342) | 22.8 / 37.6 (167 of 342) |
  | layer split of two, greedy, 512 | 21.5 / 39.2 (1071 of 1386) | 23.1 / 57.8 (1089 of 1285) | 23.2 / 53.8 (1090 of 1282) |
  | layer split of two, seeded, 96 | 21.7 / 25.0 (150 of 399) | 23.2 / 33.5 (137 of 378) | 23.3 / 43.4 (183 of 297) |
  | layer split of two, seeded, 512 | 21.4 / 31.6 (957 of 1722) | 23.1 / 35.2 (749 of 1558) | 23.2 / 50.2 (1054 of 1386) |

  With drafts on llmx leads on one MI50 and on the layer split of two in every cell at both precisions, and on the group of two at 512 tokens and in both seeded cells.
- **Open against the reference:** a group of two, greedy, 96 tokens, an ordinary chat reply: 40.4 at `int8` and 47.9 at default precision against 52.7. At `int8` the arm fed 109 drafts where every other fed over 300, the pass price holding drafts back over the first requests (32.6, 40.4 and 43.7 tok/s over the three), which the one-card and layer-split arms did not do; what to look at is the price's first requests on a group, where the sums make the first passes dear.
- **Not the drafter's:** the group of four is behind in every cell with drafts as without (20 to 21 against 49 to 51 without drafts); that is the decode cell of that width in the status table, with its recovery, measured here before the member threads landed. The drafter's own gain on a group of four is up to 2.4 times.
- **The seeded cells at `int8`** keep fewer of their drafts than at default precision on every shape, one device included (732 of 1453 against 1058 of 1374 on one MI50 at 512 tokens), while the greedy cells keep the same. Drafts on give the ids of drafts off at both precisions, so the kept rate is the numerics', the target's probability of the drafter's token under 8-bit activations, and not a defect; the reference's seeded rate (957 of 1722) sits where `int8`'s does. A profile of that probability at both precisions would say more; it is not a gate.
- **Width 1 against main** (761a38286, `llmx 0.1.0+g761a38286ff5`, against the branch before its record was completed, `llmx 0.1.0+ga9990d5d8da1`, the same src, both built the same way from detached worktrees): ids and logits the same on the CPU (Qwen3-0.6B and Qwen3.5-0.8B Q8_0), on one MI50 (those and Qwen3-8B Q8_0) and on a layer split of two (Qwen3.5-0.8B Q8_0, Qwen3.5-9B Q4_K_M); CTest 40 of 40 and the CPU suite whole; one MI50, main, branch, branch, main twice: Qwen3-8B Q8_0 pp512 834.6 to 843.7 against 834.5 to 840.9 and tg128 75.2 to 77.2 against 75.5 to 77.2 tok/s, Qwen3.5-9B Q4_K_M pp512 773.3 to 779.6 against 772.8 to 778.3 and tg128 79.8 to 81.1 against 78.8 to 81.1.
- **Reviewed by:** F2DEV at fcdb62b23 and across the rebases, the same src: the head read whole accepted as first support on the condition the cards met, the fit fix's failing test asked for as the first commit and given.
- **Left:** the many-user mix with the drafter over a group runs on CPU groups only (`spec`); on cards it is not run.
- **Gotchas:** a member holds the head twice where the file's head is not tied, its shard and the whole matrix (1.35 GiB of Q8_0 more on Qwen3.6-27B at any width), and reads the whole matrix for a draft row, so a draft row's head costs one device's time.

## The hybrid qwen35 models on a tensor group of four cards (2026-10-08, measured, docs only, lands by fast-forward)

- **Goal:** the device measurement at width 4 the hybrid models' record left pending (below).
- **Correctness, four MI50s under one root as one group** (main 6f71114ca, `llmx 0.1.0+g6f71114ca449`, default precision, every command's first line naming the group of four): the tensor-split fixtures against HF within 5.6e-7 logits (bound 2e-5) and 4e-8 NLL (bound 1e-5) and against one card by the device-reference criterion; Qwen3.5-9B Q4_K_M and Qwen3.6-27B Q4_K_M against their layered goldens, 41 of 41 each; drafts by lookup on against off on Qwen3.6-27B Q8_0, the same ids greedy and seeded, 20.3 against 29.1 tok/s greedy and 20.2 against 23.9 seeded over 96 tokens. The 16k check ran once for the family, at width 2.
- **Speed, recorded** (Qwen3.6-27B Q8_0, the same four cards, default clocks, four whole cores with nothing else on them, `bench --model --p 512 --n 128 --r 3`, two rounds interleaved with the reference's ROCm tensor split in the same shape, its log showing the groups formed, tok/s):

  | Qwen3.6-27B Q8_0, four MI50s | pp512 | tg128 |
  |---|---:|---:|
  | llmx group of four, `--dtype int8` (the matched precision) | 618.7, 618.7 | 21.0, 20.7 |
  | llmx group of four, default precision | 502.9, 501.3 | 21.1, 20.9 |
  | reference tensor split of four | 377.6, 550.3 | 50.6, 50.5 |
  | llmx two stages of two, `int8` | 629.9, 630.3 | 33.1, 32.8 |
  | reference two stages of two (`-tps 2`) | 589.5, 589.6 | 33.1 (its second round's row was not captured) |
  | llmx layer split of four, `int8` | 402.9, 402.2 | 21.3, 21.5 |

- **Open against the reference at the same precision:** decode on a group of four, 21 against 50.5 tok/s, the cell Qwen3-32B has open at that width for the same reason, a decode sum's cost on one recording thread; recovery, the member threads branch and then a backend whose submissions do not go through the kernel per sum. The prompt cells are met at both shapes, and two stages of two decode level with the reference's. The reference's two prompt rounds on a group of four, 377.6 and 550.3, are 46 percent apart in one arm of one session; the spread is the reference's and unexplained, the comparison rests on its higher round, and a third round would say which is the odd one.
- **Measured before the member threads landed** (`A group's members submit side by side`, above), which raises a group of four's decode on Qwen3-32B Q8_0 from about 21.5 to about 29.7 tok/s; the hybrid model's width 4 decode cell is to be read again at that head.

## The recorder rule holds a read that allocates its host memory (2026-10-09, branch test/read-slabs-wait, tests only, lands by fast-forward)

- **Why:** the wait before a read from disk takes its host memory (`alloc_host` in the scheduler's read) was held by no case: the recorder rule's read step finds idle slabs, which is no call on a device, and the call made without its wait passed that case ten runs of ten (The recorder rule held by a parked recording, below, What it does not hold).
- **Done:** a fifth step in `server-resume`'s recorder-rule case. A second scheduler under the same root, on a fresh model and with a host tier of 64 KiB, smaller than an entry, adopts the kept entries; with a recording parked, the follow-up of a conversation on disk is read back through host memory beyond the tier, whose slabs are taken at the read. No call may reach the stage's devices while the recording is parked, and once it is let go the scheduler's first call there must be that allocation. The steps share one body (`hold`).
- **Why not the first read after a start with a whole tier:** a start measures the disk through host memory and keeps those slabs idle within the tier, so the first read back finds them and allocates nothing; tried, its first calls there were small allocations and copies.
- **Planted, in a Windows build:** the read's `alloc_host` called without the wait fails the new step five runs of five, each with 2 calls met while the recording was parked; unplanted it passes.
- **What it still does not hold:** `trim_host` and a boundary's state alone, as before.

## The recorder rule held by a parked recording (2026-10-08, branch test/recorder-rule, tests only, lands by fast-forward)

- **Why:** the rule that a device a stage's thread records on takes no call from another thread but a wait rested on the review's list for every call but the timing read; its first test caught an unguarded call only by chance and was parked (`A timed server over stages of groups`, below).
- **Done:** `server-resume` holds it by a latch. Its backends park a stage's recording inside the stage's submission, on two stages of CPU groups with two passes in flight, a host tier of two copies and a disk tier; with the recording parked the test sends the request that makes the scheduler act, waits 300 ms, requires that no call reached that stage's devices from another thread and that the scheduler's thread made none, lets the recording go and requires that the scheduler's calls arrive, the first of them the one the step is for. The steps: a follow-up whose room sends a donor to host memory (first call a copy into host memory), a conversation promoted from host memory (first call the copy out of it), and a conversation read back from disk and promoted.
- **Planted, in a local build, two runs each, the same result both times:** `save_host` called without the wait fails the first step with 6 calls met; `restore_host` without it fails the promotion's step with 6; unplanted it passes. Under ThreadSanitizer on the test machine, where the scheduler is several times slower, `save_host` planted fails the same step with the same 6 calls inside the hold, so the sanitizer job's run of the case reads as the native one does. The schedule is the same run to run: six conversations in turn leave two copies in host memory and the oldest on disk alone.
- **What it does not hold, and why:** a step holds the first call its path makes on the stage's devices, so a path's later calls are held only where a step begins with them; `alloc_host` before a read from disk takes idle slabs there, which is no call on a device, and planted without its wait it met nothing; `trim_host` and a boundary's state alone are not reached; the timing read stays with `server-passes-cpu`'s timed case. The hold is a time, 300 ms, against a scheduler that reaches the call in well under a millisecond: a slower scheduler would let a fault through, never fail a correct build.
- **Gates** (tests only; the head before the rebase onto the member threads, 24dca2f5, and the rebased head): on the test machine a CPU build without warnings, CTest 40 of 40, `server-resume`'s CPU half three times, docs and dead-code as the list, the linked check as the list, and `server-passes-cpu` and `server-resume` under ThreadSanitizer with no report, the latter again at the rebased head in 596 s; on Windows under MSVC the CPU half passes in 113 s. After the rebase onto the disk tier's change of the same day, which rewrote what the tier writes, the case gave the same schedule and the same result on Windows, clean and against both planted faults. No suite component reads the file.
- **Reviewed by:** F2DEV at 24dca2f5, no defect and two notes, taken: the planted fault run once under ThreadSanitizer, and the test's comment saying that its first-letter checks assume idle slabs.

## Disk writes write what changed (2026-10-08, branch feat/disk-increment-3, steps 1 to 3 of DISK-TIER's Entries written as what changed, lands by fast-forward)

- **Why (the user, 2026-10-07):** disk writes should write what has changed. A conversation was one file, its whole copy, 5.2 GB at 76k tokens of a 27B model, written again whenever it had grown by a third and once more at the stop.
- **Done:** a history on disk is a path of immutable files, segments of its blocks cut at multiples of 1024 tokens and a state where each turn ended, each named by a digest of the tokens and row classes below it, so a turn writes its new blocks and its state and nothing a second time, a longer history stands on its shorter self's files, and an edit shares every segment below its fork. `src/server/disk_index.hpp` (new) is the tree with no model, store or file in it; `DiskTier` owns the index, the write in flight and the reads; the scheduler names a history by its digests and asks. The earlier steps are on main: the range copies and spans of `Model` (`ddd047213`) and the store's runs at offsets under a layout (`aeafa7b64`).
- **Removed in the same change:** the rule that spaced a conversation's idle writes by a quarter of its length, the list of whole entries with its superseding on disk, and the copy of a whole device donor through host memory for a write. `src/server/scheduler.hpp` goes from 3017 lines to 2760; `disk_tier.hpp` from 294 to 633, `disk_index.hpp` is 363.
- **Every state is kept** (the user, 2026-10-07: fast edits remain): each turn's state stays on disk within the cap and the age limit, about 150 MiB a turn on Qwen3.8-27B.
- **The layout's version is 2** (`DiskStore::kVersion`), named in the identity, so the first server of this change adopts nothing from the one before: its disk cache starts empty once.
- **As built, against the design:** a file's header holds digests and no tokens; a boundary's state is written whatever of its history's blocks is on disk, kept while the server runs and dropped at a start if its blocks never came; the design's fault cases, the order that takes a regenerated reply's old tail first, the separate counters of segments and states and the line a write were not built and are listed under Planned in DISK-TIER.
- **The failing test first:** `server-resume`'s `disk_increment` on main `24d89d62` stops at "turn 2 written while idle within a minute", the quarter rule writing nothing for a turn that added a block; with the change the bytes written equal the bytes on disk after each of four turns on the dense and the hybrid model, the stop writes nothing, and a second scheduler forks the whole conversation and an edit at its third message with the replies of a fresh model. `disk-index` (new CTest) holds the tree alone.
- **Measured** on two MI50s, Qwen3.8-27B Q8_0 with production's flags (`--device vulkan:0,vulkan:1 --drafter embedded --disk-cache-bytes 107374182400 --disk-cache-keep --disk-cache-max-age 24h`), one chat of 76k tokens over ten turns, each followed by the idle writes, the cache on the raidz1 pool of six SATA SSDs:

  | after turn | tokens reused of the prompt | first token (s) | files | written so far (GB) | on disk (GB) |
  |---|---|---|---|---|---|
  | 1 | 0 of 76018 | 274.0 | 76 | 5.541 | 5.541 |
  | 2 | 76160 of 76188 | 1.23 | 76 | 5.541 | 5.541 |
  | 3 | 76160 of 76225 | 1.85 | 78 | 5.705 | 5.705 |
  | 4 | 76224 of 76258 | 1.23 | 78 | 5.705 | 5.705 |
  | 5 | 76224 of 76282 | 1.25 | 80 | 5.870 | 5.870 |
  | 6 | 76288 of 76305 | 1.21 | 80 | 5.870 | 5.870 |
  | 7 | 76288 of 76355 | 1.84 | 82 | 6.035 | 6.035 |
  | 8 | 76352 of 76413 | 1.28 | 84 | 6.199 | 6.199 |
  | 9 | 76416 of 76483 | 1.87 | 86 | 6.364 | 6.364 |
  | 10 | 76480 of 76560 | 1.84 | 88 | 6.528 | 6.528 |
  | an edit of turn 3's question | 76160 of 76224 | 1.39 | 90 | 6.697 | 6.697 |
  | the stop | | 0.57 s, nothing left to write | 90 | 6.697 | 6.697 |

  A turn that crosses a block writes that block and its state, 0.165 GB, and one that does not writes nothing; written and on disk are equal at every row, so no byte was written twice. The 90 files are 74 segments of 1024 tokens (69 MiB), one of 384 tokens, six of one block (6 MiB with its header), the edit's of two blocks, and eight states (151 MiB).
- **Against the quarter rule** (the record of 2026-10-07 below, the same chat): ten turns wrote 6.53 GB where it wrote 6.57 GB, the same within a turn's state, since over ten short turns that rule never came to its rewrite; what is gone is the rewrite of the whole copy, 5.2 GB each time the conversation has grown by a third and at every stop after a turn (6.2 s there, 0.57 s here), and with it the quarter of a conversation a crash could lose. First-token times are the same, 1.2 to 1.9 s a turn after the first.
- **A start on the kept directory:** the 90 files were in the index 0.14 s after the store was made (1.5 s after the server answered). The next turn reused 76544 of 76589 tokens with its first token after 10.1 s, 8.2 s of it the read of the 82 files of its path, 5233 MiB, about 670 MB/s; one file a conversation read at 852 MB/s on production, so the path of files costs about a quarter more time on a read this long. An edit of turn 2's question then reused 76160 of 76188 tokens in 3.0 s, reading one state, 150 MiB, since host memory held the rows.
- **The index's cost** (asked in review): with 1008 files and 50 histories asked about, on the test machine's EPYC 7262, a use of a file cost 3.0 ms, the rebuild of every link, on the scheduler thread; the links are now kept across a use, which changes none, and only what follows from the uses is rebuilt, 0.03 ms. A file that comes or goes still rebuilds them, 3 ms, and the scheduler then asks about its 50 histories again, 1.0 ms, once a file written or deleted. A history's digests at 76k tokens take 0.8 ms, once a request.
- **A wait on the scheduler thread** (noted in review): an idle write of a device donor copies one file's bytes off the devices and waits for that copy under the lock, 3.7 ms a segment of 1024 tokens (4.8 at most) and 7.1 ms a state, and the call takes 6.6 ms in all on average and 45 ms where it first takes its slabs; a timing build on the same chat's first three turns and an edit, 79 files. It runs only while the server is idle or stopping, so a request that arrives then waits that long at most.
- **Gates** against main `6f71114ca`, before the rebase onto `761a3828` and the review's two changes to the index: Qwen3-0.6B Q8_0 and Qwen3.5-0.8B Q8_0 give the same ids and logits as main on the CPU and on one MI50; the suite on the MI50 passes, `server` included, and `server-resume` with its device half; the host tier's eviction run on Qwen3.5-0.8B Q8_0 gives main's replies with first tokens at a median 0.037 s in both; CTest and the CPU suite pass on Linux and Windows; `bench` on the MI50, Qwen3-0.6B Q8_0, two rounds interleaved, reads a 512-token prompt at 8960.6 and 8964.2 tok/s against main's 8981.4 and 8952.2 and generates at 399.1 and 401.9 against 401.4 and 401.8.
- **Left:** steps 4 and 5 of the design (Planned in DISK-TIER), and the read of a long path, which one reader thread takes file by file.
- **Reviewed:** by F2DEV: the ownership is the one agreed, the five departures from the design each accepted, and the failing test holds the bytes written to the bytes on disk. It asked for the index's cost at production's size and noted the wait on the scheduler thread, both above; `DiskIndex::find`, which only the test called, went when the linked check named it, and the delta was read again before landing.

## A group's members submit side by side (2026-10-08, branch perf/tp-member-threads, lands by fast-forward)

- **Why:** width 4's decode is bound by the one thread that records a group: at every sum it made each member's queue call and sync-file calls in turn, and the profile had that thread about 80 percent in the kernel, half of it in the submission (`A decode sum at width 4`, below). The calls of different cards do not depend on each other.
- **Change:** `flush` is three steps, `prepare` (the command buffer ended, the ticket taken), `Submission::queue` (the queue's call alone) and `queued` (the ring advanced, a failure thrown). At a sum the recording thread prepares every member, the queue calls and the exports of the sync files run side by side, the first member's on the recording thread and each other's on a thread of the collective, and the recording thread takes every queue back, makes the imports and only then throws. A member's thread calls its device's queue and its semaphores' exports and nothing of the backend's state; between `prepare` and `queued` the backend refuses to record or submit. The threads are the scheduler's sampling pool, moved unchanged to `core/job_threads.hpp` (`core::JobThreads`): they sleep between jobs, are woken once a job, and the caller takes every index no thread has woken for. Only a group of three or four has them.
- **Measured before building** (timing builds, never landed; Qwen3-32B Q8_0, `--dtype int8`, a group of four MI50s under one root, default clocks, four whole cores): an exchange took 285 to 413 us with the calls in turn, 171 to 211 us with them side by side and 180 to 215 us where each member also imported its own waits behind a mutex and condition variable, so that form was not built. Side by side a member's queue call takes 62 to 69 us against 44 alone, and the imports left on the recording thread 42 to 57 us an exchange; those two are why the result is below the 35 to 36 tok/s a spinning barrier reached in a timing build, which was rejected for what it does on fewer cores than members.
- **Result, tg128 in tok/s, the same model and cards, both arms built the same way** (main 24d89d62 `llmx 0.1.0+g24d89d624575`, the branch `llmx 0.1.0+g5cdf6b14c94b`, and a control: the branch's code with the member threads off, so main's behaviour in the branch's layout), three rounds in turn: main 21.7, 21.2, 21.6; control 20.9, 21.1, 21.5; branch 29.8, 29.5, 29.9; pp512 588 in every arm. Greedy ids are main's at width 4, on two stages of two and on a group of two (48 tokens each). A group of two has no thread and is level: 29.9 and 29.7 against 29.8 and 29.7; with threads it measured 30.1 against 29.8 and the same serving rates for a fifth more CPU, which is why it has none.
- **Cores** (the process's CPU time from its cgroup over 15 s of one request generating, and the rate of that run; main, then the branch):

  | placement, whole cores given | cores busy | tok/s | CPU ms a token |
  |---|---|---|---|
  | a group of four, four | 0.99, 0.99 against 1.89, 1.97 | 21.7, 21.5 against 28.9, 30.6 | 45, 46 against 66, 64 |
  | a group of four, two | 0.87 against 1.96 | 22.2 against 29.4 | 39 against 67 |
  | two stages of four on eight MI50s, eight | 0.99 against 1.88 | 20.4 against 29.2 | 49 against 65 |
  | two stages of four, four | 0.99 against 1.97 | 21.0 against 29.1 | 47 against 68 |
  | two stages of four, two | 0.99 against 1.95 | 21.3 against 29.8 | 47 against 66 |

  On one whole core a group of four measured 25.2 and 26.2 against main's 15.4 and 22.0, the spread of one shared core. It was never below main on any core count. An idle server's threads sleep: no CPU.
- **Serving, 128-token prompts, 128 generated, greedy, no request failed** (output tok/s, inter-token p99 in ms, the server's CPU ms an output token; main, then the branch). A group of four on four whole cores at 1 / 4 / 8 / 16 / 32 users: 18.5 / 70.3 / 124.2 / 161.3 / 177.3 against 24.4 / 89.8 / 141.8 / 161.7 / 178.0; p99 58 / 54 / 57 / 718 / 965 against 40 / 40 / 43 / 717 / 964; CPU 52 / 12.9 / 6.7 / 3.7 / 2.1 against 73 / 18.3 / 9.7 / 5.0 / 2.8. So the gain is where few requests generate, up to 8 users; from 16 users the rate is the same for about a third more CPU a token. Two stages of four on eight MI50s at 32 users on eight, four and two whole cores: 310.3, 311.3, 307.7 against 310.6, 312.8, 311.0; p99 784, 787, 786 against 691, 785, 787; CPU 3.9, 3.8, 4.2 against 5.3, 5.4, 5.5. Two stages of two, prompts of 64 to 1024 tokens at 1 / 4 / 16 / 32 / 64 users, twice an arm: 18.6 / 55.5 / 120.7 / 122.8 / 115.7 and 18.6 / 55.0 / 121.1 / 124.4 / 121.6 against 18.6 / 55.2 / 121.9 / 124.9 / 119.5 and 18.4 / 55.5 / 120.5 / 124.2 / 115.4.
- **Open against the reference at the same precision:** a group of four at one user, 24.4 against the reference's 43.2 tok/s of the same-topology session (bench 29.8 against 49); the status table has the recovery.
- **Tests:** `vulkan-lifetime` on four MI50s, five runs of five passing: a sum whose first and whose last member's queue call fails on whichever thread makes it, on a group of three (its collective has the threads), the files this process holds open counted around every failed sum, a collective destroyed right after its failed sum, and the planted fault, a backend whose queue a prepared submission holds refusing to record and to submit. `backend-vulkan`'s sums on two and three devices, both sides of the two-shot size, unchanged. No hosted runner has three cards, or two, so the three-device cases and the collective's are a hand check, as AGENTS says of the collective; the open files are counted on Linux, where the system lists them, and not elsewhere.
- **Activity during the hand runs, as posted by the others in the collaboration log:** a CPU build on logical CPUs 0-3 and 8-11 for the first nine minutes of the eight-card runs (the eight-core cell's two cores-busy runs), a build of under a minute on the same CPUs, and a container that raised the load average to 108 for a minute before the 1 and 4 user runs of the two-rep serving cell; the mixed load on two stages of two was taken again after a run with neighbours on its cards, the first set aside, not used.
- **Gates** (the branch 5cdf6b14 against main 24d89d62, both built the same way from detached worktrees; then rebased onto main 6f71114c with no conflict, the code under `src/backends`, `src/core` and `src/server` unchanged by it): on the MI50 machine CTest 45 of 45, the suite whole on the CPU and on one MI50 but for two stale paths in `docs/src`, corrected here; byte identity of greedy ids, the last logits row and perplexity at width 1 on the CPU and on one MI50 for Qwen3-0.6B, Qwen3-8B and Qwen3.5-0.8B Q8_0 in each load mode (9 runs, 144 lines each), and of greedy ids of Qwen3-32B Q8_0 at `int8` on a group of four, two stages of two and a group of two. One timing round on one MI50, Qwen3-8B Q8_0, main, branch, branch, main twice: pp512 843.3 to 853.9 against 845.4 to 850.2 and tg128 77.6 to 77.8 against 77.5 to 77.8 tok/s; the group's round with its control is above. The CPU's synthetic bench on one thread, beside another session on the sibling threads: prefill 5599 to 5620 against 5534 to 5633 tok/s, decode 4962 to 5285 against 4471 to 5230, read as level within that neighbour's spread.
  On the Radeon VII under the AMD driver on Windows, where no group forms and the change is `flush` in three steps: CTest 46 of 46, ids, 64 logits rows and a 4096-token perplexity window of Qwen3-0.6B Q8_0 and Qwen3-8B Q8_0 byte for byte, and three rounds in turn, pp512 3184 to 3191 against 3186 to 3190 and tg128 204.5 to 205.1 against 204.0 to 205.4 on the 0.6B, pp512 345.8 to 347.6 against 346.1 to 347.5 and tg128 42.3 to 42.4 against 42.3 to 42.4 on the 8B.
  The 16k long-context checks were not run: the change makes no arithmetic, and byte identity is its check.
- **Reviewed by:** F2DEV at 69c19ab7, no defect and three notes, taken here: the hand check named with its runs, the open files said to be counted on Linux alone, and the comment on the collective's threads cut to its pointer. Its design review and its ACK of the form built are in the collaboration log.

## The tensor split runs the hybrid qwen35 models (2026-10-07, branch feat/tp-qwen35, the tensor split's step 5, lands by fast-forward)

- **Goal:** `--tensor-width N` on Qwen 3.5, 3.6 and 3.8, which a group refused for their linear-attention layers' recurrent state (`docs/TENSOR-SPLIT.md`, step 5).
- **Done:** the refusal is gone and the runtime carries a state over a group: each member allocates the state of its own K and V heads (`shard::state`) and its marks' saved rows at its share of each width (`shard::saved`, `shard::saved_floats`, which the fit counts too), `group_prepare` gives every member the slots the group shares, `group_record` saves a marked entry's inputs on each member, and `rerun` runs each member's layers over what it saved with its own weights; the qwen35 module reads its member's V heads from the step's width; `check_plan` refuses by name a saved row a width does not divide. Nothing changed in a backend, the collective, the scheduler or the server, and a model at width 1 runs the code it ran. Drafting by lookup works over a group with this, as do state checkpoints and the host and disk tiers, each member's state slot copied on its own.
- **What a group splits and what it does not:** `docs/TENSOR-SPLIT.md`, Step 5 as built. Every real file forms widths 2 and 4; width 3 is refused on each by its KV heads and its K heads, so no uneven shares are needed for this family.
- **Correctness on CPU groups** (hosted, every job): the qwen35 fixture of the tensor-split component at widths 2 and 4 against HF, max logit error 4.2e-7 against the F32 bound of 2e-5 and NLL error 4e-8 against 1e-5 (the dense fixtures included in those maxima), and against one backend by the device-reference criterion; `llmx-split-check` on that fixture as two stages of groups of two against one group, bit-identical over the prompt, slices of 1, 3 and 16, decode steps, passes in flight, the recompute by class and the verifies of drafts with their retracts; `spec` on a hybrid model whose shape a group of two splits whole (`kHybridEven`): every proposer and sampler against the run without drafts on one group, and 300 tokens of marks, verifies of 2 to 17 rows and retracts to every kept count against single steps, on one and on two stages of groups; `server-resume`: pauses and resumes, a fork at a checkpoint, a paused request's state kept and taken back, donors to host memory and back member by member; `server-passes-cpu`: the paused, held and steady loads on 1, 2 and 3 stages of groups at every P, and a drafting scheduler at two passes over two stages of groups, where a stage's thread saves a marked entry's inputs, which ran under ThreadSanitizer on the test machine with no report; `shard`: a member's state with a mark against a hand count.
- **Correctness on two MI50s as one group** (under one root, RADV, default precision, 4b1c3f359 before its rebase, `llmx 0.1.0+g4b1c3f35921f`; every command's first line `tensor group vulkan:0+vulkan:1 under` its root, the witness that it ran under the width):

  | check at `--tensor-width 2` | result |
  |---|---|
  | the tensor-split fixtures against HF | max logit error 5.2e-7 (bound 2e-5), NLL error 8e-8 (bound 1e-5) |
  | the suite's baseline (Qwen3-0.6B Q8_0, Q4_0, Q5_K_M, Q4_K_M), server and chat components | pass, every NLL cell inside its bound |
  | Qwen3.5-0.8B Q8_0 against its HF goldens (`tests/baseline_qwen35.py`) | 59 of 59 |
  | Qwen3.5-9B Q4_K_M against the layered goldens (`tests/baseline_layered.py`) | 41 of 41 |
  | Qwen3.6-27B Q4_K_M against the layered goldens | 41 of 41 |
  | drafts by lookup on against off, greedy and seeded, Qwen3.5-9B Q4_K_M and Qwen3.6-27B Q8_0 (`generate`, 96 tokens) | the same ids |
  | the many-user mix on Qwen3.5-9B Q4_K_M with log-probabilities (`tools/server_mix_check.py`) | 0 of 8 differ together, 0 of 6 skewed, the CLI's text |
  | its uncapped phase, 12 requests of 3455 to 4021 tokens on a pool of 4096 | 0 of 12 differ through 23 pauses and 15598 recomputed tokens |
  | the 16k check on Qwen3.6-27B Q8_0 against one card (`tools/long_context_check.py`) | two fresh servers the same 512 tokens; one card's top choice at 512 of 512, largest gap 0.000 logits |

- **Width 1 against main** (362398d11, `llmx 0.1.0+g362398d11a63`, both built the same way from detached worktrees): ids and logits the same on the CPU (Qwen3-0.6B and Qwen3.5-0.8B Q8_0), on one MI50 (those and Qwen3-8B Q8_0) and on a layer split of two (Qwen3.5-0.8B Q8_0, Qwen3.5-9B Q4_K_M); CTest 40 of 40 and the CPU suite whole. One MI50, main, branch, branch, main twice on a quiet machine: Qwen3-8B Q8_0 pp512 836.1 to 848.1 against 831.8 to 842.8 and tg128 77.3 to 77.7 against 77.1 to 77.6 tok/s; Qwen3.5-9B Q4_K_M pp512 774.5 to 777.6 against 771.8 to 777.9 and tg128 80.9 to 82.0 against 80.3 to 81.5. An earlier round under a machine load of 20 read the 9B 1.5 percent lower on the branch in its second half and level in its first. No perturbed control build was run: the quiet round is within a percent both ways, and the only change on a width 1 path is a division of a head count by the step's width.
- **Speed, recorded at first support** (Qwen3.6-27B Q8_0, two MI50s under one root, default clocks, four whole cores with nothing else of this work on them, `bench --model --p 512 --n 128 --r 3`, two rounds, tok/s; the reference is its ROCm tensor split of two on the same cards, its log showing the group of two formed, taken ten minutes after the llmx arms and not between them):

  | Qwen3.6-27B Q8_0 | pp512 | tg128 |
  |---|---:|---:|
  | llmx group of two, `--dtype int8` (the matched precision) | 646.1, 645.6 | 33.6, 33.3 |
  | llmx group of two, default precision | 426.5, 426.3 | 33.4, 33.9 |
  | reference tensor split of two | 631.0, 630.6 | 36.6, 36.7 |
  | llmx layer split of two, `int8` / default | 412.4, 412.2 / 262.6, 262.5 | 23.5, 23.5 / 23.4, 23.4 |
  | llmx one card, `int8` / default | 417.8, 417.6 / 264.9, 264.7 | 23.8, 23.9 / 23.7, 23.7 |

  One round each at `int8`: Qwen3.8-27B Q8_0, group of two 646.3 and 33.4 against its layer split 412.6 and 23.5; Qwen3.5-9B Q4_K_M, group of two 1751.5 and 97.6 against one card 1178.3 and 93.6.
- **Open against the reference at the same precision:** decode at width 2, 33.4 against 36.6 tok/s on Qwen3.6-27B Q8_0, the cost of a decode sum, the cell Qwen3-32B has open too (the tensor split's records below); the prompt cell is met, 646 against 631.
- **Width 4 on cards** was measured after this landed, in the record above (The hybrid qwen35 models on a tensor group of four cards).
- **Gotchas:** the numerics fingerprint changes with the model layer's files, so a kept disk cache of an earlier build is not adopted. A member's arena and its rerun's room keep one device's slot widths, larger than its shards need, and the fit counts them as allocated.
- **Reviewed by:** F2DEV at 4b1c3f359: the state's ownership over a group right; two findings taken, the saved row's refusal in `check_plan` and a test that puts `save` on a stage's thread under ThreadSanitizer, the second in `server-passes-cpu`, which the job runs, since `spec` makes no scheduler.

## A timed server on the CPU keeps a host stage's time from its first pass (2026-10-07, branch fix/timed-cpu, lands by fast-forward)

- **Found** by the health redesign's work and reproduced on main: `llmx serve --timing` on a CPU-only server closed the connection at its first request with no error. A timed scheduler adds a host stage's own time to a per-stage list as the stage is recorded, and sized that list only at the end of the first round; on the CPU the first pass's stage runs on the host before any round has ended, so the first add wrote past an empty list and the process died. A device's stage is not on the host and never added there, which is why every timed run on cards worked.
- **Fix:** the list is sized with the scheduler. Failing test first: `server-passes-cpu` runs a timed scheduler on one CPU and split over two; at the test's commit it ends in a segmentation fault in three runs of three, at the fix every reply is its reply alone and the timing holds its rounds and a time a stage.
- **Scope:** `--timing` with a stage on the host, so a CPU-only server or experts on the CPU; an untimed server never reached it.
- **Reviewed by:** F2DEV at d0968007, one finding, taken: the list is still zeroed where the first span starts, as the device's reading is discarded there, and the test holds every stage's time finite and at least 0.

## STATUS by month: September's records moved to their own file (2026-10-07, branch docs/status-archive, docs only, lands by fast-forward)

- **Done:** 96 dated records and historical blocks older than 2026-10-01, 10083 lines, moved unchanged and in order to [STATUS-2026-09](STATUS-2026-09.md); STATUS.md went from 11824 to about 1740 lines. Nine "Left: nothing" lines of the kept records were removed and four reworded to the fact they held; the Markdown that named a moved record names the archive, and the status table's rows whose record moved link to it.
- **Left:** nothing; the code's comments that named the moved records now name the archive, and the docs check reads the archive as records because its title is dated; at each month's end that month's records move the same way.

## Docs tightness: repeated text kept once (2026-10-07, branch docs/tightness-h, docs only, lands by fast-forward)

- **Done:** the `--dtype` table is kept in USAGE and PRECISION points to it; the disk tier's ranking and read bounds are stated once in DISK-TIER and SERVER and `docs/src/server.md` refer to them; `docs/CI.md` counts the suite's 28 components from `tests/run_tests.py`; and `docs/DISK-TIER.md` no longer claims the server refuses a `--disk-cache-dir` whose ACL grants others access, which no code does. On Windows `owner_only` sets nothing and no ACL is read, so the directory and its files inherit the ACL of `--disk-cache-dir`; USAGE, `docs/src/server.md` and `docs/src/format-file_writer.md` say owner-only for POSIX alone. Writing the check is the alternative.
- **Left:** `docs/CI.md` still holds the first hosted runs of the five-job workflow (the run on `08351b0` and the repair `851d375`, near the end of the page's first section), history of the same kind that the audits did not name.

## History sentences moved out of live docs (2026-10-07, branch docs/tightness-h, docs only, lands by fast-forward)

- **Done:** the sentences below were measurements and history in pages that describe how things are now; they are kept here as they stood, and the pages say what is true now or point to the records.

- From docs/USAGE.md, the tensor split section:
  > On two MI50s a group of two decodes about as fast as one card on a model one card holds and reads a prompt 1.6 times as fast, and decodes a model too large for one card faster than a layer split; serving many users, one group of two is below a layer split of the same two cards at default precision from 16 users on.
  > Stages of groups overlap when several passes are in flight: on four MI50s, Qwen3-32B Q8_0 as two stages of two at `--dtype int8` served more than a layer split of four at every user count from 1 to 64, and at default precision up to 16 users, about level or below beyond; on eight MI50s two stages of four are below a layer split of eight from 4 users on.

- From docs/MULTI-DEVICE.md, Tensor split:
  > - The group sum is deterministic: partial sums added in a fixed member order on every member. Earlier, an 8-card reduction that added peers in per-rank order silently diverged at 100k context, and a reused scatter region raced when message sizes grew; both are design constraints here.

- From docs/DISK-TIER.md, Why:
  > On the six-user, twenty-turn workload of Qwen3.8-27B Q8_0 on one MI50 ([STATUS](STATUS.md), the edited-turn gap after message boundaries), a conversation's copy holds about 800 MiB at turn 19 and a message boundary 150 MiB, and the default host tier, half of what the host has free, holds about six conversations and four boundaries each.
  > Whatever falls out of host memory is recomputed: about 4.4 ms a token, so 43 s for a 9.7k-token conversation, and the edited-turn gap is mostly boundaries the host tier had no room for (2.54 s at a 16 GiB host tier against 3.31 s at 10 GiB).

- From docs/DISK-TIER.md, Demotion: host to disk:
  > the clause "as a first version of this plan had it," after "Demoting only at the moment of need," (the sentence goes on: keeps nothing, because the entry handed to the writer still holds its slabs)

- From docs/CI.md, the optional 8B consumer:
  > At `dacf18c` local Windows and Linux runs each passed the 37 checks the consumer had before the per-token half, with identical printed NLLs and HF deltas.
  > The Linux ordinary suite passed its 11 components with `--no-perf-floor` at that commit.
  > These local results do not establish hosted 8B coverage; the optional consumer is not run by the workflow.

## The tensor split's kernel route is closed (2026-10-07, the user's decision, docs only, lands by fast-forward)

- **Done:** `docs/TENSOR-SPLIT.md`, section 8, records the decision and its two reasons: the route needs a patched kernel and a patched Vulkan driver, which a release cannot ask of users, and it cannot be tested on the machine that hosts production without risk. The readings and the two candidate patch sets stay as a record under headings that say closed. The status table's open decode cells name their recovery as the member threads branch and then a backend whose submissions do not go through the kernel per sum, the planned ROCm backend after the tensor split is complete on Vulkan.
- **Left:** nothing here. The question of where to test the route, which earlier records list as waiting on the user, is answered by the closing; those records are not rewritten.

## Llmx against the reference in the same topology (2026-10-07, measured, docs only, lands by fast-forward)

- **Why:** the user's rule is the same topology on both sides, each arm at its best fair settings. Earlier serving tables here put llmx's two stages of two or its layer split beside the reference's one tensor group of four; the notes under those records say which of their cells were the same shape.
- **Setup, as the measurer posted it (2026-10-07 14:15 UTC):**
  The user's rule: the same topology on both sides, each arm at its best fair settings. Qwen3-32B Q8_0, MI50s under one root (GPU[2] to GPU[5]; the group of two on GPU[2]+GPU[3]), default clocks, one session, every server on logical CPUs 4-7 and 12-15 (cores 4 to 7 whole), the three arms of a topology and load in turn from fresh servers, the order rotated a load. llmx is main 6ccb6352 (`llmx 0.1.0+g6ccb6352f88d`, with the hold once a stage) at `--dtype int8`, the matched precision, and at default precision; the reference is `llama-server` of the pinned image (eefc4e732) with its image environment and GPU_MAX_HW_QUEUES=8 from its recommended settings, `-ngl 99 -fa 1 -lm dio -cram 0`, and for the shape: two stages of two `-sm tensor -tps 2`, a group of four or of two `-sm tensor -tps 0`, a layer split `-sm layer`. Its logs at a verbose level, taken before the run, show the shape formed: "creating a Meta device for tensor parallelism from 4 devices (tps=2, n_stages=2)", its custom all-reduce initialized for each group of 2, devices 0 and 1 loading from layer 0 and devices 2 and 3 from layer 32, and "pipeline parallelism enabled" for the staged and the layer split alike; n_batch 2048, n_ubatch 512. Pools: 81920 tokens over 64 sequences for the closed loads, 131072 over 20 for the skewed one, on every arm. 128 generated tokens, greedy. No request failed in any arm that started.
  One arm did not start: the reference's group of two with the skewed load's pool (`-c 131072 -np 20` on two cards) ended at load with "failed to allocate buffer for kv cache"; its row is missing and nothing stands in for it.
  Other sessions' containers ran on other cards and other logical CPUs through the session (the monitor has them every ten seconds); the machine's load average was 5 to 25.

- **Tables, copied from the post** (output tok/s, time to first token p50, inter-token p50 and p99, failed; `llmx --dtype int8` is the matched arm by the user's rule, `llmx default` is what a user who does not pass `--dtype` gets, and both stay in the table):

```
== two stages of two; users 1 / 4 / 16 / 32 / 64: output tok/s | ttft p50 s | itl p50 ms | itl p99 ms | failed
128  llmx --dtype int8  | 27.6 / 93.8 / 233.9 / 290.2 / 260.9 | 0.26 / 0.87 / 1.66 / 2.70 / 4.82 | 34 / 36 / 49 / 74 / 129 | 37 / 37 / 57 / 948 / 1117 | 0
128  llmx default       | 26.9 / 80.3 / 173.1 / 180.8 / 200.6 | 0.40 / 1.28 / 2.38 / 3.88 / 7.00 | 34 / 40 / 64 / 126 / 202 | 36 / 41 / 71 / 1371 / 1671 | 0
128  reference          | 28.7 / 76.1 / 106.2 / 163.0 / 193.1 | 0.35 / 1.16 / 3.74 / 4.11 / 9.65 | 32 / 42 / 120 / 136 / 213 | 61 / 97 / 161 / 246 / 3582 | 0
1024 llmx --dtype int8  | 19.3 / 51.5 / 82.6 / 89.3 / 88.5 | 2.02 / 3.56 / 9.83 / 18.17 / 38.73 | 36 / 39 / 58 / 93 / 168 | 38 / 522 / 1108 / 1147 / 1220 | 0
1024 llmx default       | 16.4 / 40.9 / 59.7 / 60.3 / 61.5 | 3.14 / 5.16 / 13.96 / 26.78 / 54.63 | 37 / 43 / 74 / 144 / 241 | 39 / 738 / 1566 / 1668 / 1773 | 0
1024 reference          | 22.1 / 43.9 / 50.8 / 57.7 / 57.8 | 1.57 / 5.10 / 14.34 / 26.99 / 55.13 | 32 / 44 / 125 / 145 / 228 | 65 / 87 / 3134 / 3657 / 4282 | 0
mix  llmx --dtype int8  | 19.8 / 56.3 / 121.7 / 124.7 / 125.5 | 1.77 / 3.40 / 7.64 / 10.26 / 22.77 | 37 / 38 / 55 / 90 / 163 | 39 / 108 / 1068 / 1169 / 1281 | 0
mix  llmx default       | 17.4 / 44.3 / 85.6 / 81.9 / 84.4 | 2.75 / 5.01 / 10.18 / 15.34 / 33.53 | 36 / 43 / 71 / 141 / 235 | 38 / 153 / 1755 / 1845 / 2033 | 0
mix  reference          | 22.9 / 43.9 / 58.8 / 72.2 / 78.3 | 1.35 / 4.58 / 12.85 / 24.20 / 38.60 | 33 / 44 / 124 / 143 / 220 | 65 / 103 / 1119 / 4405 / 4405 | 0
   skew: users tok/s | users itl p50, p99 ms | longest gap s | gaps over 0.5 s | long prompts ttft s | long tok/s | failed
   llmx --dtype int8  |  128.9 |   49,  1388 |  2.81 |  328 of 8128 | 9.9 to 20.8 |  15.7 | 0
   llmx default       |   96.6 |   65,  1864 |  2.52 |  350 of 8128 | 13.7 to 28.9 |  11.0 | 0
   reference          |   61.1 |  211,  3101 |  4.69 |  164 of 8127 | 11.6 to 29.0 |  10.2 | 0
== one group of four; users 1 / 4 / 16 / 32 / 64: output tok/s | ttft p50 s | itl p50 ms | itl p99 ms | failed
128  llmx --dtype int8  | 20.2 / 70.1 / 160.5 / 177.9 / 203.0 | 0.24 / 1.09 / 2.15 / 3.99 / 7.77 | 47 / 48 / 70 / 124 / 185 | 58 / 61 / 717 / 965 / 1056 | 0
128  llmx default       | 20.5 / 69.7 / 122.3 / 138.3 / 148.2 | 0.32 / 1.26 / 2.66 / 4.99 / 9.84 | 46 / 47 / 95 / 160 / 258 | 53 / 54 / 910 / 1225 / 1357 | 0
128  reference          | 43.2 / 104.9 / 141.0 / 195.2 / 177.0 | 0.34 / 1.14 / 4.24 / 6.22 / 12.31 | 20 / 28 / 80 / 100 / 216 | 44 / 68 / 121 / 157 / 4335 | 0
1024 llmx --dtype int8  | 16.2 / 37.8 / 52.0 / 53.3 / 52.2 | 1.77 / 4.69 / 15.64 / 30.78 / 69.49 | 48 / 48 / 80 / 144 / 280 | 51 / 930 / 972 / 1023 / 1136 | 0
1024 llmx default       | 15.2 / 33.2 / 40.7 / 42.2 / 42.3 | 2.25 / 5.75 / 19.85 / 39.09 / 81.45 | 49 / 48 / 106 / 180 / 353 | 53 / 1170 / 1234 / 1299 / 1447 | 0
1024 reference          | 27.7 / 42.1 / 44.4 / 49.3 / 49.7 | 1.99 / 7.07 / 21.23 / 37.98 / 72.24 | 20 / 29 / 82 / 102 / 218 | 47 / 213 / 4297 / 4423 / 4390 | 0
mix  llmx --dtype int8  | 16.6 / 42.3 / 80.7 / 78.4 / 81.3 | 1.50 / 4.14 / 11.12 / 16.56 / 38.42 | 48 / 48 / 77 / 142 / 224 | 56 / 908 / 1007 / 1071 / 1188 | 0
mix  llmx default       | 16.0 / 37.3 / 61.4 / 61.2 / 62.8 | 1.91 / 5.13 / 14.19 / 21.71 / 46.40 | 48 / 48 / 104 / 178 / 296 | 59 / 1147 / 1352 / 1464 / 1552 | 0
mix  reference          | 29.4 / 47.3 / 67.7 / 75.5 / 75.8 | 1.72 / 5.95 / 10.38 / 25.96 / 40.92 | 20 / 29 / 81 / 100 / 212 | 47 / 79 / 3835 / 4507 / 4583 | 0
   skew: users tok/s | users itl p50, p99 ms | longest gap s | gaps over 0.5 s | long prompts ttft s | long tok/s | failed
   llmx --dtype int8  |   92.5 |   70,  1155 |  1.21 |  613 of 8128 | 8.9 to 34.7 |  10.0 | 0
   llmx default       |   71.0 |   97,  1412 |  1.44 |  632 of 8128 | 12.2 to 44.3 |   7.7 | 0
   reference          |   70.3 |  164,  4240 |  4.69 |  147 of 8127 | 16.8 to 35.2 |  10.0 | 0
== one group of two; users 1 / 4 / 16 / 32 / 64: output tok/s | ttft p50 s | itl p50 ms | itl p99 ms | failed
128  llmx --dtype int8  | 28.2 / 85.7 / 152.4 / 167.7 / 153.7 | 0.26 / 1.15 / 2.38 / 4.48 / 8.96 | 33 / 38 / 73 / 126 / 249 | 36 / 40 / 826 / 1116 / 1248 | 0
128  llmx default       | 27.2 / 72.8 / 96.5 / 110.5 / 107.9 | 0.40 / 1.55 / 3.36 / 6.47 / 12.98 | 34 / 43 / 120 / 196 / 389 | 36 / 44 / 1198 / 1626 / 1825 | 0
128  reference          | 30.6 / 80.0 / 106.7 / 167.7 / 192.1 | 0.33 / 1.09 / 3.94 / 5.89 / 8.52 | 30 / 41 / 119 / 131 / 206 | 47 / 69 / 155 / 162 / 3638 | 0
1024 llmx --dtype int8  | 19.3 / 36.6 / 45.5 / 45.8 / 43.9 | 1.99 / 5.27 / 17.83 / 35.42 / 80.34 | 36 / 44 / 93 / 171 / 346 | 39 / 1075 / 1129 / 1229 / 1407 | 0
1024 llmx default       | 17.1 / 28.1 / 30.9 / 31.5 / 30.8 | 2.92 / 7.47 / 26.13 / 51.71 / 113.69 | 36 / 49 / 140 / 241 / 471 | 38 / 1531 / 1648 / 1772 / 1999 | 0
1024 reference          | 22.2 / 39.1 / 42.7 / 48.5 / 50.1 | 1.85 / 6.64 / 19.66 / 36.34 / 70.49 | 30 / 42 / 123 / 139 / 222 | 48 / 71 / 4102 / 4301 / 4338 | 0
mix  llmx --dtype int8  | 20.3 / 40.4 / 70.8 / 68.0 / 67.1 | 1.69 / 4.66 / 11.57 / 19.60 / 42.86 | 36 / 43 / 87 / 162 / 321 | 38 / 1026 / 1218 / 1232 / 1459 | 0
mix  llmx default       | 18.0 / 32.0 / 47.4 / 46.3 / 45.3 | 2.50 / 6.64 / 16.92 / 28.79 / 64.19 | 36 / 48 / 134 / 232 / 459 | 38 / 1493 / 1846 / 1990 / 2191 | 0
mix  reference          | 23.2 / 42.5 / 60.0 / 67.9 / 76.1 | 1.61 / 4.90 / 10.60 / 25.66 / 39.06 | 30 / 42 / 122 / 139 / 220 | 47 / 75 / 3570 / 4567 / 4423 | 0
   skew: users tok/s | users itl p50, p99 ms | longest gap s | gaps over 0.5 s | long prompts ttft s | long tok/s | failed
   llmx --dtype int8  |   76.5 |   76,  1480 |  1.68 |  648 of 8128 | 10.1 to 41.8 |   8.3 | 0
   llmx default       |   52.4 |  124,  2011 |  2.08 |  648 of 8128 | 15.3 to 60.2 |   5.6 | 0
   reference          missing
== layer split of four; users 1 / 4 / 16 / 32 / 64: output tok/s | ttft p50 s | itl p50 ms | itl p99 ms | failed
128  llmx --dtype int8  | 16.6 / 63.0 / 195.6 / 288.2 / 262.6 | 0.41 / 1.25 / 2.04 / 2.96 / 4.68 | 57 / 54 / 61 / 76 / 124 | 59 / 54 / 62 / 100 / 2085 | 0
128  llmx default       | 15.8 / 57.0 / 155.4 / 193.2 / 169.4 | 0.70 / 2.01 / 3.25 / 4.78 / 7.29 | 58 / 55 / 69 / 111 / 225 | 60 / 58 / 70 / 141 / 2571 | 0
128  reference          | 15.3 / 50.4 / 73.4 / 119.4 / 133.2 | 0.48 / 1.64 / 6.22 / 8.19 / 13.98 | 54 / 67 / 170 / 185 / 313 | 193 / 71 / 174 / 195 / 5015 | 0
1024 llmx --dtype int8  | 11.9 / 42.4 / 83.4 / 98.5 / 102.1 | 2.89 / 4.15 / 9.59 / 16.57 / 32.94 | 62 / 57 / 71 / 95 / 161 | 64 / 58 / 1784 / 1823 / 1890 | 0
1024 llmx default       | 10.2 / 34.5 / 60.5 / 66.4 / 62.9 | 4.73 / 6.50 / 14.65 / 25.51 / 56.41 | 62 / 58 / 79 / 129 / 259 | 63 / 59 / 2724 / 2786 / 2927 | 0
1024 reference          | 14.3 / 31.9 / 41.4 / 46.1 / 42.3 | 2.03 / 6.10 / 15.67 / 30.67 / 68.78 | 55 / 70 / 179 / 202 / 348 | 58 / 74 / 3706 / 4938 / 6715 | 0
mix  llmx --dtype int8  | 12.5 / 45.2 / 112.5 / 130.8 / 140.9 | 2.48 / 3.67 / 7.67 / 10.54 / 21.36 | 61 / 57 / 68 / 91 / 154 | 62 / 58 / 1248 / 1877 / 2095 | 0
mix  llmx default       | 10.6 / 36.9 / 79.4 / 86.3 / 90.5 | 4.15 / 5.84 / 12.17 / 17.40 / 33.25 | 62 / 57 / 76 / 125 / 253 | 64 / 71 / 1905 / 3314 / 3269 | 0
mix  reference          | 14.6 / 30.2 / 45.6 / 53.8 / 54.5 | 1.83 / 6.38 / 14.75 / 32.64 / 55.14 | 55 / 70 / 175 / 195 / 336 | 58 / 75 / 3650 / 6222 / 6633 | 0
   skew: users tok/s | users itl p50, p99 ms | longest gap s | gaps over 0.5 s | long prompts ttft s | long tok/s | failed
   llmx --dtype int8  |  119.7 |   61,  2100 |  3.59 |  204 of 8128 | 16.9 to 18.7 |  16.2 | 0
   llmx default       |   91.7 |   70,  3082 |  4.16 |  213 of 8128 | 25.3 to 27.7 |  11.5 | 0
   reference          |   52.2 |  229,  3587 |  5.83 |  147 of 8127 | 13.0 to 32.4 |   8.3 | 0
```

- **Read, llmx at int8 against the reference in the same shape** (output tok/s at 1 / 4 / 16 / 32 / 64 users), as posted:
- **Two stages of two:** ahead from 4 users on under every load (128-token prompts 93.8 / 233.9 / 290.2 / 260.9 against 76.1 / 106.2 / 163.0 / 193.1; 1024-token 51.5 to 89.3 against 43.9 to 57.8; mixed 56.3 to 125.5 against 43.9 to 78.3) and behind at one user under every load (27.6 against 28.7, 19.3 against 22.1, 19.8 against 22.9). Under skew the users get 128.9 tok/s against 61.1. Inter-token p99 is behind at 32 users on short prompts (948 against 246 ms) and at 4 users on 1024-token prompts (522 against 87), ahead at 64 users and from 16 users on the longer loads.
- **One group of four:** behind at 1 and 4 users under every load (20.2 against 43.2 and 70.1 against 104.9 on short prompts) and at 32 users on short prompts (177.9 against 195.2); ahead at 16 and 64 users on short prompts and from 16 users on the longer loads; under skew 92.5 against 70.3.
- **One group of two:** behind at one user under every load and at 64 users under every load (153.7 against 192.1, 43.9 against 50.1, 67.1 against 76.1), behind at 4 users on the longer loads and at 32 users on 1024-token prompts; ahead at 4 and 16 users on short prompts and at 16 on the longer ones; level at 32 users on short and mixed prompts. No reference row under skew.
- **Layer split of four:** ahead at every user count on short prompts (16.6 / 63.0 / 195.6 / 288.2 / 262.6 against 15.3 / 50.4 / 73.4 / 119.4 / 133.2) and from 4 users on the longer loads; behind at one user on 1024-token and mixed prompts (11.9 against 14.3, 12.5 against 14.6); under skew 119.7 against 52.2.
- At default precision llmx is behind the reference in more cells than at int8 in every tensor shape; the tables carry both.
- **Contention:** the measurer's account above stands: other sessions' containers on other cards and logical CPUs, the monitor's samples every ten seconds, a load average of 5 to 25. No arm was discarded or rerun for it, so the figures are read with that flag and not as idle-machine figures.
- **Open speed cells, recorded in the status table:** one user behind the reference in every tensor shape at int8; the group of four at 1 and 4 users; a group of two at 64 users; inter-token p99 at 32 users on short prompts. A path main already has is behind in them, so they are open and not waived.
- **Not measured:** the reference's group of two under the skewed load (its pool did not fit two cards), and a reference arm at any other precision.

## `/v1/health` in groups, `/v1/live`, grouped help pages and the operating guide (2026-10-07, branch feat/health-help, lands by fast-forward)

- **Goal:** a health reply and help pages a person can read, and a page for people who run a server (`docs/OPERATING.md`).
- **Done:** `/v1/health` is nested groups (`server`, `precision`, `requests`, `reuse` with `device`, `host`, `disk` and `boundaries`, `pressure`, `reread`, `drafting`, `passes`, `timing` under `--timing`), counters split into `now` and `since_start` with units in the names; the mapping from every old field is in `docs/SERVER.md` (The grouped `/v1/health`) and every field is in a table in `docs/USAGE.md`. New values: version, numerics, uptime, context, devices, the limits, the two tier caps, the sampling threads, `in_flight` of the disk tier and three request totals (three atomic counters in `Scheduler::finish`). `GET /v1/live` answers a constant without the scheduler's lock. The `serve` page is in five groups (Server, Limits, Prefix cache, Speculative decoding, Execution) with three examples, the `generate` and `chat` pages in Generation, Sampling and Speculative decoding groups, and the pages' plain words replace "routed layers" and "passes in flight". No flag was renamed, added or removed and no default changed. Every reader in the tree moved with it: `tests/server.py`, `tests/qwen35.py`, `tests/drafters.py`, `tests/server_mix_tool.py`, `tools/server_load.py`, `tools/server_mix_check.py`. `tests/server.py` pins the groups, the totals and `/v1/live`.
- **Left:** outside the tree, the operator's own readiness polls and the A/B gate scripts that read the flat names move to the new ones. `bench` keeps its one group.
- **Gotchas:** `--timing` on a CPU server closed the connection at the first request with no message, on main as well as here (checked on main `63354caa`); it was not changed. The `finished` total counts requests that were admitted, so one cancelled while queued is not in it.

## An idle server does not rewrite a conversation every turn (2026-10-07, branch feat/idle-rewrite-rule, lands by fast-forward)

- **Why (the coordinator for the user, 2026-10-07):** the idle write under `--disk-cache-keep` wrote a conversation's whole copy after its turns, 5.2 GB at 76k tokens of a 27B model, where a turn adds a tenth of that; on the user's pool that is real wear.
- **Done:** an idle server leaves a conversation alone while an earlier whole copy of it on disk is within a quarter of its new length (`Scheduler::near_on_disk`), and under keep that earlier file is no longer deleted when a later turn's copy supersedes it but once a newer copy's file is in place; room takes such a file first. A clean stop writes everything, as before. The test commit fails without it: a turn that grew a conversation from 1152 to 1280 tokens wrote its whole copy again.
- **A hole this closes:** before it, the turn after a written one deleted the conversation's file, its copy superseded, and wrote none, the superseded copy in host memory counting as the newer one's; so after each turn that left its checkpoint in the same block as the turn before, three of the ten below, a conversation had no copy on disk and a crash then lost it whole. The table's "on disk" column shows it at turn 2.
- **Measured** on two MI50s, Qwen3.8-27B Q8_0 with production's flags, one conversation of 76k tokens over ten turns, each followed by the idle writes, then a stop:

  | after turn | written so far, main (GB) | on disk, main (GB) | written so far, this branch (GB) | on disk, this branch (GB) |
  |---|---|---|---|---|
  | 1 | 5.62 | 5.62 | 5.62 | 5.62 |
  | 2 | 5.62 | 0.16 | 5.62 | 5.62 |
  | 3 | 11.24 | 5.78 | 5.78 | 5.78 |
  | 5 | 16.87 | 5.95 | 5.94 | 5.94 |
  | 7 | 22.51 | 6.11 | 6.10 | 6.10 |
  | 10 | 39.43 | 6.60 | 6.57 | 6.57 |
  | the stop | 0.6 s, nothing left to write | | 6.2 s, the newest copy written, the older file deleted | |

  39.4 GB against 6.6 GB over the ten turns; what this branch still writes a turn is the turn's message boundary, 0.16 GB. Time to first token is the same in both, 1.3 to 1.9 s a turn after the first.
- **Cost:** a crash loses at most the last quarter of a conversation's length, which its next turn reads again after forking the earlier copy; the stop writes a copy main had already written, 6 s here.
- **Planned, not built:** an entry written as what changed (`docs/DISK-TIER.md`, Order of work), which is a redesign of the entry and not a change in the store.
- **Reviewed** by F2DEV, no finding; its notes taken: after a crash that left two copies of one history the next server marks the shorter superseded again, held by a test; DISK-TIER says the rule is by a history's tokens and names the window in which the age limit or a tight cap can take the earlier file before a newer copy exists.

## The age-limit check asks about a file being deleted without throwing (2026-10-07, branch fix/age-exists, lands by fast-forward)

- **Found:** `server-resume` failed in a hosted Windows job with "exists: Access is denied" on an entry file, on a branch that does not touch the age limit; seen once before in 96 loaded runs on a Windows PC.
- **Cause:** the check, not the server. While the age limit deletes the entries the check counts those still there with `std::filesystem::exists`, which on Windows throws for a file whose deletion is pending.
- **Done:** it asks with an error code and counts a file that refuses the question as still there, so it asks again until the file is gone. Tests only.
- **Reviewed** by the coordinator.
- **Lesson:** this was seen once in a load run the day before and only noted; a failure seen once is posted as a finding the same hour.


## A kept entry's identity follows the numerics, not the revision (2026-10-07, branch feat/numerics-identity, lands by fast-forward)

- **Goal (the user, 2026-10-06):** an update should keep the conversations on disk unless it changes what their bits depend on. The identity carried the build's version string, so every update of production started with an empty disk tier, four times in one day.
- **Done:** the identity is the digest of components a line each: the model file's digest, a numerics fingerprint, the compiler's version text, the compiler, configuration, flags and options the build system used, the shader compiler's version, the C library where it names itself, and the model's host layout with each device's driver. The revision is not among them.
- **The fingerprint** is the SHA-256 over every file under `src/`, `cmake/`, `CMakeLists.txt` and `build.bat` that `cmake/numerics-sources.txt` does not name out, CRLF read as LF. In by default: the list names out the CLI, the hub, the server, the tokenizer and the files that only parse, sample or print, each with its reason, and allows eight includes of such files from inside with theirs. Both build routes compute it at every build with no Git; a Linux build with GCC and a Windows checkout with CRLF line endings give the same value at one commit, as does `build.bat`.
- **Held by `tests/version.py`:** the binary's fingerprint is the one the test computes from the tree; every `src/` file is in or named out; every line names a path that exists; no file inside includes an out file that no allow line argues; and seven planted faults.
- **Said by the server:** its start line prints the fingerprint, `llmx --version` too, the components are left as text in the server's directory, and a server that adopts nothing from a kept directory names the components that differ.
- **Measured** on two MI50s, Qwen3.8-27B Q8_0 with production's flags, a 76k-token conversation written by this branch's build and its server stopped:

  | restarted on | fingerprint | adopted | next turn: reused of 76194 tokens | first token (s) | start line |
  |---|---|---|---|---|---|
  | a build that changes one server file | the same | 3 of 3 entries | 76160 | 7.7 | 3 entries adopted |
  | a build that changes one kernel file | another | none | (not sent) | | a kept directory was not adopted, its numerics differing from this server's |

- **Not covered, and said in DISK-TIER:** the C library on Windows and macOS is unrecorded; a file argued out of the list wrongly would let an entry be read against other bits, which the review of each line and the include rule bound, and no test proves.
- **The idle rewrite's cost (asked with this branch):** USAGE now gives it measured, 5.2 GB a turn at 76k tokens, 260 GB a day at fifty such turns. Writing only what a turn added is not a small change: an entry is one file, written once, laid out layer by layer, and the index, room, age, superseding and adoption all take an entry as a whole. A proposal is in the devlog; nothing is built.
- **Reviewed** by F2DEV: every out and allow line held against the code. Taken from the review before landing, since a later addition to the identity empties every cache once more: a Vulkan device's line carries its pipeline cache id, which changes with every build of the driver (held by reading, a rebuilt driver not being something the test machines can fake); the flags carry the interprocedural-optimization setting; the layout names the tensor width; an entry file's bytes and the entries' descriptions are held to digests at `DiskStore::kVersion`; `disk-store` holds that the build's facts carry the configuration; a kept directory from before this branch says why it gave nothing.


## Where an edit's first token goes (2026-10-06, measured at main `737b5fa4`, docs only)

- **Question (the user):** edits were behind the reference although checkpoints and rollbacks are what the tiers are for.
- **Measured:** Qwen3.8-27B Q8_0 on one MI50, the aged-conversation load (every user's conversation taking turns, then a regenerate and an edit of an earlier message each), llmx with host and disk tiers at its default precision and at `--dtype int8`, and the reference runtime, all on the same card in the same container image, at 6 and at 24 users. Per edit, medians: the tokens read after the fork, the prompt time a token read, the time queued, and the time to the first token.

  | users | arm | edits | tokens read | prompt (ms/token) | queued p50 / max (ms) | edit first token p50 / p99 (s) | regenerate p50 (s) | follow-up p50 / p99 (s) |
  |---|---|---|---|---|---|---|---|---|
  | 6 | llmx, default (16-bit activations) | 6 | 450 | 4.50 | 158 / 319 | 2.17 / 3.02 | 2.12 | 2.27 / 3.77 |
  | 6 | llmx, `--dtype int8` | 6 | 450 | 2.84 | 190 / 194 | 1.49 / 1.94 | 1.54 | 1.56 / 2.45 |
  | 6 | reference | 6 | 411 | 3.89 | 0 / 0 | 1.62 / 1.94 | 5.15 | 2.57 / 4.36 |
  | 24 | llmx, default (16-bit activations) | 24 | 362 | 4.81 | 2 / 452 | 1.85 / 3.22 | 2.70 | 2.49 / 39.97 |
  | 24 | llmx, `--dtype int8` | 24 | 380 | 3.05 | 3 / 220 | 1.10 / 1.97 | 3.37 | 1.39 / 2.92 |
  | 24 | reference | 24 | 344 | 4.08 | 0 / 0 | 1.30 / 1.96 | 4.90 | 7.24 / 30.06 |

- **Reading:** the fork itself costs nothing measurable; an edit's wait is the tokens it reads after the boundary times the prompt rate, plus at 6 users a wait behind a re-read job's pass. At the precision the reference computes in, `--dtype int8`, llmx's edit is ahead at both levels (1.49 against 1.62 s, 1.10 against 1.30 s); at the default 16-bit activations it is behind by the prompt rate, which is the precision choice and not the tiers.
- **Two candidates, neither built (the coordinator, 2026-10-06):** letting a request's prompt pass go ahead of a re-read job's pass, worth 0.16 to 0.32 s an edit at 6 users; and a boundary at the exact token in place of the whole block below it, worth 0.11 to 0.18 s, which needs a partial block's rows kept with the state and a fork of a partial block, neither of which exists.
- **A limit of the tiers as sized, in the default arm at 24 users:** 11 follow-ups of turns 17 to 19, conversations of 8.6k to 10.1k tokens, reused nothing and read their whole prompts in 37 to 44 s, which is that arm's p99. Each was admitted without a wait (no read from disk was tried, no request was paused, the read wait bound never fired), so its conversation's copy was in neither host memory nor on disk; the same users' turns before and after reused as usual. What differs from the int8 arm, which had none: the default arm's server was given 4759 MiB of host memory for the tier against 6453 MiB (half of what the host had free at each start) the KV pool being 22592 tokens in both, so about six copies of such a conversation fit its host tier against eight, and of the bytes moved to host memory 51 percent reached the disk against 67 percent. The disk tier stood at its 64 GiB cap in both arms. The likely rule is therefore the host tier's room taking a copy before its write to disk had come up, one write being in flight at a time; the server did not count copies dropped unwritten or entries deleted for the cap when this ran, so this run cannot tell the two apart; `/v1/health` now counts both (`reuse.disk.since_start.lost_before_written` and `reuse.disk.since_start.dropped_for_cap`), and a rerun of this arm with them would. The reference reused nothing on any follow-up at 24 users (456 of 456), its slots being fewer than the users.
- **Open:** the 24-user rerun of the default arm with those two counters has not been run, so which rule dropped the 11 follow-ups' copies is not known.
- **Conditions:** one run an arm, each on a fresh server, the arms one after another on the same card at its default clocks; a monitor recorded the machine every 10 s (its load average, 5 to 33 over all three arms of the 24-user run, with other sessions' builds on other cores, the busiest processes, the card's use and memory) and no arm was dropped for it; the p99s are nearest rank. The run's files are on the test machine in the measurer's scratch directory, `mb/ed1` (6 users) and `mb/ed2` (24 users): each arm's per-request records, serve log, health and monitor.
## An idle server writes ahead under keep, and the stop's flush states its bound (2026-10-06, branch feat/keep-idle-flush, lands by fast-forward)

- **Goal:** a stop under `--disk-cache-keep` should find little left to write, say how long it will take, and say what it did not write; and a conversation larger than the stopping server's host tier should be kept at all, which fix/keep-large left open.
- **Done:** a server with nothing active, queued or paused for five seconds writes what the flush would, one entry at a time, host memory's entries first and then the device donors; a donor's copy larger than the host tier is held beyond the tier until its file is in place, at a stop and while idle alike. The flush's bound is 20 seconds or, where longer, half as long again as what is left takes at the store's measured write rate plus five seconds; it prints the bytes and the bound before its first write and the entries kept and the bytes not written, with the reason, after its last. The line a start prints for adopted entries says how many each rule dropped, and `/v1/health` counts what the tiers lost while running: `host_unwritten`, copies room took from host memory before any file held them, and `disk_capped`, entries deleted for the cap. The test commit fails without the change: an idle scheduler wrote nothing.
- **Measured** on two MI50s, Qwen3.8-27B Q8_0 with production's flags and a scratch directory, a 76k-token conversation after its second turn, stopped with `docker stop -t 30` and started again into a 3 GiB host tier:

  | | written before the stop | the stop (s) | flush line | next turn after the restart: reused of 76194 tokens | first token (s) |
  |---|---|---|---|---|---|
  | stopped at once, no idle time | nothing | 7.5 | writing 5248.0 MiB, within 20 s; 0.0 MiB not written | 75968 | 9.5 |
  | left idle first | 3 entries, 5.78 GB, within 13 s of the idle moment | 0.56 | writing 0.0 MiB; 0.0 MiB not written | 76160 | 7.6 |

  The idle row needs the re-read job to end, which fix/job-forks-request made it do; before that fix the job read the conversation again for minutes and the server was never idle.
- **A limit, seen twice in these runs:** where the host is short of free memory the copy of a donor larger than the tier is refused beside the reserve the host keeps, and that conversation is then not kept; the stop says so (`a donor of 76160 tokens was not kept in host memory`, `5312.0 MiB not written`). The machine these runs share had 13 GiB free at the time.
- **What it costs:** a conversation's whole copy is written again after each turn that is followed by five idle seconds, its older file deleted when the turn's job supersedes it: 5.2 GB a turn for a 76k-token conversation on that model. A request that arrives while a donor is copied off the devices waits for that copy as it waits for an eviction's, about 2 s for 5.2 GiB here.
- **Stop grace:** the bound is 20 s for anything the store writes in 10 s, about 7 GB here; production's `docker stop -t 30` covers that, and a server holding more unwritten at a stop names a longer bound in its first line.
- **Reviewed** by F2DEV, no finding; its four notes are taken: each idle period tries every donor again, `server-resume` holds the stop's own flush, its 20-second bound under a paced write and a second idle period, `disk_capped` is said to count while running, and USAGE names the cost a turn. With them goes the nit left from fix/job-forks-request: a job that must wait for its request's pass is refused before a host donor is promoted for it.
- **Left:** the cache identity keyed on what can change the bits, so an update keeps the entries; a copy held beyond the tier whose request was cancelled goes at the first round with nothing queued or paused, not with its request.

## A tensor group's member is held once a stage, not at every sum (2026-10-07, branch perf/tp-stage-hold, lands by fast-forward)

- **Found:** a count of submissions under a RADV built with symbols showed half of a group's submissions were holds. On a placement over several devices each backend follows a submission with a second one that waits on an event, so a waiting card keeps its clock; a layer split submits once a pass a device, and a tensor group submits at every sum, 128 a token a member, each with its hold, on the one thread whose time bounds width 4's decode (`A decode sum at width 4`, below).
- **Change:** a sum's submission is followed by no hold; the stage's own submission holds as before. A layer split's stage is unchanged, one hold a pass a device.
- **Measured before building** (a timing build of main a01216fb with the hold chosen by the environment; Qwen3-32B Q8_0, `--dtype int8`, four MI50s under one root, default clocks, the process on cores 4 to 7 whole, `bench --model --p 512 --n 128 --r 3`, two rounds, interleaved; greedy ids those of main in every form; tg128 in tok/s, with the cards' share of one-second samples at the highest clock state that run reached, which compares arms of one placement and is no absolute frequency):

  | | two stages of two | one group of four | one group of two |
  |---|---|---|---|
  | main, a hold after every submission | 28.4, 28.6 (60 to 66%) | 16.2, 16.3 (83 to 87%) | 29.7, 29.6 (74 to 76%) |
  | a hold after a stage's own submission only | 29.4, 29.3 (53 to 64%) | 21.2, 21.8 (31 to 34%) | 29.7, 29.7 (69 to 76%) |
  | no hold at all | 21.3, 21.2 (2 to 8%) | 21.1, 21.2 (10 to 14%) | 29.6, 29.5 (43 to 48%) |

  pp512 is 525 to 526, 587 to 589 and 532 to 533 in every form. No hold at all was measured and rejected: a member waiting for the other stage drops its clock and the staged placement loses a quarter. Decode submissions in a run at width 4 fell from 410496 to 205256 with the holds after sums gone.
- **Open against the reference at the same precision:** width 4 decode, about 21.5 against 49 tok/s; a thread a member with one wake an exchange measured 34 on top of this in a timing build and is the next step, a design note first.
- **Gates** (the branch against main 65b40150, both built the same way from detached worktrees, `llmx 0.1.0+gd574dea8b743` and `llmx 0.1.0+g65b40150c537`; then rebased onto main with no conflict in code): on the MI50 machine, CTest 45 of 45, the suite whole on the CPU and on one MI50, and at `--tensor-width 2` its split, tensor-split, server and chat components (f32 skips an odd-shaped fixture there); byte identity of ids, logits and perplexity at width 1 on the CPU and on one MI50 for Qwen3-0.6B, Qwen3-8B and Qwen3.5-0.8B Q8_0 in each load mode, and of greedy ids and the last logits row of Qwen3-32B Q8_0 on a layer split of four, a group of two, a group of four and two stages of two, at default precision and `int8`; docs, dead-code and the linked check.
  One timing round, interleaved main, branch, branch, main twice, `int8` on the groups (tok/s): one MI50, Qwen3-8B Q8_0, pp512 831.6 to 834.5 against 829.2 to 836.9 and tg128 77.0 to 77.3 against 77.2; Qwen3-32B Q8_0 on four MI50s, tg128 on a layer split of four 16.8 to 17.2 against 16.9 to 17.4, on two stages of two 29.0 to 29.3 against 27.7 to 29.2, on a group of four 19.5 to 21.5 against 15.0 to 16.5, on a group of two 29.6 to 30.0 against 29.5 to 29.8; pp512 level in every placement (326, 525, 587, 533).
  The two-stage server load (`serve --max-seqs 64`, a pool of 81920 tokens, 128 generated tokens at 1 / 4 / 16 / 32 / 64 users, greedy, no request failed): 128-token prompts 27.5 / 96.9 / 232.6 / 287.8 / 274.1 against 26.5 / 88.8 / 226.7 / 286.4 / 252.3 output tok/s, inter-token p99 37 / 39 / 65 / 945 / 1126 against 37 / 46 / 60 / 945 / 1204 ms; prompts of 64 to 1024 tokens 19.7 / 57.1 / 121.3 / 125.3 / 123.5 against 18.5 / 52.5 / 119.3 / 125.2 / 123.2, p99 39 / 149 / 1202 / 1180 / 1272 against 45 / 152 / 1074 / 1259 / 1311.
  On the Radeon VII under the AMD driver on Windows, where no group forms: the Vulkan CTests, Qwen3-0.6B Q8_0 ids and logits byte for byte, pp512 3121 to 3135 against 3124 to 3133 and tg128 202.1 to 203.5 against 202.7 to 203.3.
- **Reviewed by:** F2DEV at 745728c0, no finding: byte identity is the check for a change that makes no arithmetic, and the 16k long-context checks were not run for that reason.

## A re-read job waits for its request's pass and forks it (2026-10-07, branch fix/job-forks-request, lands by fast-forward)

- **Found:** in production (Qwen3.8-27B Q8_0 over two MI50s) the job that reads a reply again read the whole conversation, 103744 rows after a 96209-token prompt, and the reply beside it decoded at 8.4 tok/s where the next long turn decoded at 16.0.
- **Cause, from a trace of each job's admission kept in memory and printed at the stop:** the ids a chat route gives while a reply is written reach the scheduler in whatever round they arrive, and `enter` took a running request as a job's source only where no pass held it. In a round where a pass did, the request was passed over although it gave 4544 of 4608 tokens, the donors gave nothing, and the job was admitted reading from its first token. Over a split with two passes in flight that is most rounds. A build with prints in the loop, and the suite's own case, which gives the ids from a pass's retirement, both met the request between passes and never showed it.
- **Done:** a job whose best source is in a pass is not admitted that round and forks it in the round the pass has retired in (`Scheduler::enter`). The test commit fails without it: eight requests in turn over two CPUs, each given its next turn's ids from the reader's thread, had jobs reading 384 rows where 128 lie past the request's prompt blocks.
- **Measured** on two MI50s, Qwen3.8-27B Q8_0 with production's flags, a 76018-token prompt with a 120-token reply and its follow-up:

  | | rows the job read | reply (tok/s) | job still in flight 45 s after the reply | follow-up: reused of 76158 tokens | follow-up first token (s) |
  |---|---|---|---|---|---|
  | main `4c909f26` | 13312 and rising | 8.6 | yes | 75968 | 2.86 |
  | this branch | 128 | 15.6 | no | 76096 | 0.96 |

- **Reviewed** by F2DEV.


## Prompt rows beside decoders: what `--ubatch` trades when serving (2026-10-06, measured, docs only, lands by fast-forward)

- **Same shape or not (2026-10-07):** no cell of this record compares like with like: its reference row is the reference's one tensor group of four, beside llmx's two stages of two and its layer split of four. The same-shape table is the record "Llmx against the reference in the same topology", above.

- **Goal:** the inter-token p99 cells open on stages of tensor groups, measured before any rule: what a decoder's gap is made of when prompts land, on two stages of two and on a layer split of the same four cards, across `--ubatch` 512, 256, 128 and 64.
- **Decision** (the coordinator, 2026-10-07): no scheduler rule. The figures show a trade on tensor groups and a loss on a layer split under skew, and a feature merges only with a measured gain on its workload. `docs/USAGE.md` and `docs/SERVER.md` say what the knob trades.
- **Measured** (Qwen3-32B Q8_0, `--dtype int8`, four MI50s under one root, default clocks, logical CPUs 8 to 11, every llmx server with `--timing`, a pool of 81920 tokens over 64 sequences, 128 generated tokens, greedy, fresh servers; the groups on `llmx 0.1.0+g7613411dd481`, the layer split and the reference `llama-server -sm tensor` of the pinned image on main a01216fb an hour earlier; no request failed). Another session's container was pinned to logical CPUs 0 to 3, the sibling threads of these, through both sessions, so every cell ran beside it; the layer split at ubatch 512, run in both sessions, read 282.9 and 282.9 tok/s at 32 users on the closed load and 163.2 and 134.6 under skew, which is the spread to read the skewed table with.
  A pass's time by its prompt rows (`bench --p R`): 512 rows 0.97 s on the groups and 1.57 s on the layer split, 256 rows 0.49 and 0.81 s, 128 rows 0.27 and 0.42 s, 64 rows 0.15 and 0.26 s; a prompt reads at 526 / 522 / 483 / 440 and 326 / 318 / 308 / 248 tok/s. A decoder's row rides one pass and its token comes when that pass has left the last stage; the longest gaps at 512 rows, 1.97 and 2.37 s, are about two passes. The timed servers' stages were 25 to 33 percent idle under skew in every arm.

  | 128-token prompts, users 1 / 4 / 16 / 32 / 64 | output tok/s | inter-token p99, ms | time to first token p50, s |
  |---|---|---|---|
  | groups, ubatch 512 | 25.0 / 83.7 / 213.3 / 277.3 / 277.2 | 39 / 47 / 56 / 946 / 1148 | 0.26 / 0.94 / 2.23 / 2.94 / 5.09 |
  | groups, 256 | 25.0 / 80.8 / 217.2 / 273.1 / 279.6 | 40 / 63 / 604 / 832 / 630 | 0.26 / 0.65 / 1.78 / 2.79 / 5.04 |
  | groups, 128 | 25.0 / 81.3 / 219.6 / 273.5 / 265.2 | 41 / 44 / 396 / 335 / 389 | 0.26 / 0.74 / 1.61 / 2.72 / 5.31 |
  | groups, 64 | 25.1 / 81.4 / 214.3 / 262.5 / 252.1 | 39 / 101 / 249 / 212 / 432 | 0.27 / 0.75 / 1.72 / 3.10 / 6.32 |
  | layer split, 512 | 15.9 / 61.0 / 189.4 / 282.9 / 300.7 | 61 / 59 / 68 / 104 / 1766 | 0.42 / 1.26 / 2.28 / 2.97 / 4.79 |
  | layer split, 256 | 15.7 / 63.1 / 187.9 / 286.9 / 284.5 | 62 / 59 / 147 / 877 / 940 | 0.43 / 0.90 / 2.14 / 2.52 / 4.41 |
  | layer split, 128 | 15.6 / 65.3 / 187.6 / 286.8 / 279.2 | 62 / 61 / 479 / 495 / 542 | 0.42 / 0.57 / 1.92 / 2.35 / 4.39 |
  | layer split, 64 | 15.9 / 65.9 / 175.9 / 273.2 / 274.7 | 61 / 62 / 336 / 301 / 511 | 0.40 / 0.54 / 2.22 / 2.58 / 5.01 |
  | reference tensor split | 43.2 / 105.3 / 144.1 / 200.9 / 190.2 | 44 / 62 / 117 / 147 / 4094 | 0.34 / 1.13 / 4.17 / 6.22 / 10.69 |

  | prompts of 64 to 1024, users 1 / 4 / 16 / 32 / 64 | output tok/s | inter-token p99, ms | time to first token p50, s |
  |---|---|---|---|
  | groups, ubatch 512 | 17.8 / 51.3 / 117.5 / 121.5 / 118.8 | 44 / 112 / 1115 / 1304 / 1472 | 1.93 / 3.63 / 7.86 / 11.00 / 22.81 |
  | groups, 256 | 17.9 / 52.3 / 113.1 / 118.0 / 117.4 | 44 / 540 / 695 / 711 / 850 | 1.93 / 3.39 / 7.00 / 10.79 / 24.22 |
  | groups, 128 | 17.6 / 49.9 / 108.0 / 109.6 / 108.1 | 50 / 317 / 419 / 533 / 657 | 1.89 / 3.24 / 7.83 / 12.67 / 26.06 |
  | groups, 64 | 17.6 / 48.2 / 101.5 / 100.5 / 101.8 | 46 / 193 / 301 / 485 / 515 | 2.03 / 3.39 / 9.57 / 14.09 / 33.51 |
  | layer split, 512 | 11.8 / 43.8 / 103.2 / 129.6 / 126.0 | 70 / 66 / 1154 / 2001 / 2124 | 2.50 / 3.49 / 7.19 / 10.26 / 21.69 |
  | layer split, 256 | 11.7 / 43.8 / 100.6 / 118.7 / 127.9 | 83 / 91 / 1128 / 1167 / 1283 | 2.50 / 3.59 / 8.05 / 10.95 / 20.06 |
  | layer split, 128 | 11.4 / 45.8 / 97.3 / 113.4 / 121.5 | 177 / 69 / 686 / 725 / 970 | 2.50 / 3.25 / 7.18 / 11.37 / 23.53 |
  | layer split, 64 | 11.6 / 45.1 / 92.8 / 109.1 / 111.5 | 69 / 66 / 476 / 504 / 636 | 2.69 / 3.46 / 7.88 / 12.42 / 28.01 |
  | reference tensor split | 28.4 / 45.3 / 67.0 / 71.9 / 65.5 | 59 / 99 / 3714 / 4635 / 4686 | 1.74 / 5.86 / 10.89 / 26.58 / 43.27 |

  | skew: 16 users at 128/128, four 4096-token prompts arriving together ten seconds in (a pool of 131072 tokens over 20 sequences) | users' tok/s | users' inter-token p50, p99, ms | longest gap | gaps over 0.5 s of 8128 | the long prompts' first token | long tok/s |
  |---|---|---|---|---|---|---|
  | groups, ubatch 512 | 128.9 | 51, 1391 | 1.97 s | 341 | 10.3 to 21.6 s | 15.2 |
  | groups, 256 | 125.6 | 52, 799 | 1.64 s | 636 | 10.9 to 22.8 s | 14.8 |
  | groups, 128 | 114.5 | 53, 492 | 1.26 s | 80 | 12.8 to 26.5 s | 13.4 |
  | groups, 64 | 101.7 | 72, 685 | 1.89 s | 122 | 16.6 to 33.9 s | 11.1 |
  | layer split, 512, both sessions | 163.2, 134.6 | 68, 94 and 63, 86 | 1.53 s, 2.37 s | 55, 57 | 16.5 to 18.4 s | 15.6, 16.9 |
  | layer split, 256 | 151.2 | 66, 651 | 3.83 s | 85 | 17.5 to 18.7 s | 16.3 |
  | layer split, 128 | 126.4 | 64, 546 | 2.83 s | 97 | 19.2 to 20.1 s | 15.5 |
  | layer split, 64 | 123.2 | 63, 293 | 1.72 s | 36 | 19.7 to 20.3 s | 15.5 |
  | reference tensor split | 64.3 | 190, 4367 | 4.47 s | 148 of 8127 | 13.1 to 35.8 s | 9.7 |

- **Open against the reference at the same precision:** the inter-token p99 at 32 users on 128-token prompts on stages of groups, 946 against 147 ms, 335 at 128 rows; and at 4 users on mixed prompts, 112 against 99. From 16 users on mixed prompts, at 64 users and under skew llmx's p99 is a third or less of the reference's at the default.
- **Candidate, not built:** a cap of 128 prompt rows on a pass that carries decoders, on tensor groups alone. Its own measurement would have to show what the global knob cannot: that a prompt beside nobody keeps its 512 rows and its rate, that the p99 at 16 users on short prompts does not rise as it does with the knob, and that the 32-user cell closes rather than narrows, at an output cost below the knob's.

## A kept entry larger than the host tier is read back (2026-10-06, branch fix/keep-large, lands by fast-forward)

- **Found:** in production (Qwen3.8-27B Q8_0 over two MI50s) an 80k-token chat took minutes to its first token after a restart. Reproduced on two other MI50s at main `737b5fa4` with production's flags: a 76k-token conversation's follow-up reuses 75968 tokens before a stop (3.3 s); at `docker stop -t 30` its donor on the cards is copied to host memory and written, 5.45 GB in a stop of 7.9 s, and the same build adopts it; the next turn then reuses nothing and reads 76194 tokens in 275 s, with no read started. A 21k-token conversation survives the same sequence (21312 tokens reused, 3.5 s).
- **Cause:** `Scheduler::start_read` refused, without a word, an entry whose bytes exceed the host tier, and the restarted server's default tier was 3044 MiB (half of the host's free memory at start, the file just written having filled the file cache) against the entry's 5194 MiB.
- **Done:** such a copy is read through host memory beyond the tier, promoted and released, its file kept; where the host cannot hold it the server says so. The test commit fails without it: a second scheduler whose host tier is smaller than an entry reads none of three kept entries and its follow-ups reuse nothing.
- **Its slabs** neither count against the limit the tier's own copies are allocated within (`parked_held`) nor stay in the model's idle pool once the copy has gone (`Model::trim_host`, which the test holds: no slab left after the requests). Without the first, a conversation whose reply had been read again, which leaves a message boundary beside its copy, lost the boundary's read to the copy's (`no host memory for the copy within its limit`) and read its 76194 tokens anew in 276 s; no fixture model's copy exceeds one 64 MiB slab beside a tier of one, so no hosted test reaches that, and the hand check is the two-card sequence with the stop made after the reply was read again. Without the second the process held the tier and the entry, 5.2 GiB here, for its life.
- **Reviewed** by D2CDEV (no finding at `5f37c4d6`) and F2DEV (the idle pool, a check that could not fail, and a superseded copy beyond the tier counted as the tier's room; all three fixed).
- **Left:** this is the read side only. A stop still writes no donor larger than the stopping server's own host tier (`write_back` refuses it), so under a tier smaller than its copy a conversation is kept by neither stop, and one read back survives exactly one restart; the next branch, feat/keep-idle-flush, writes such a donor through host memory too. A copy read beyond the tier whose request was cancelled stays until the first round with nothing queued or paused, not until its request is gone. Production's own loss that day was the build identity, which changes at every update and follows as its own branch (the user, 2026-10-06). The default host tier's swing with the file cache, 3 to 21 GiB between two starts here, is recorded and not changed.

## A timed server over stages of groups reads its device times with the recorders idle (2026-10-06, branch fix/timed-groups, lands by fast-forward)

- **Found:** `llmx serve --timing` over two stages of tensor groups died at its first request with a corrupted heap (main a01216fb, Qwen3-32B Q8_0 on four MI50s, every such server of a measurement; the same servers without `--timing` and a timed layer split ran). A timed scheduler reads each stage's device time at the end of a round, a call on that stage's devices from the scheduler's thread, and since the stage threads a recorder may be recording there; the one call another thread may make on a recording backend is a wait on a returned ticket. The host copies, marks, retracts and kept states were routed through the wait for idle recorders when the threads landed, and this reading was missed.
- **Fix:** both readings go through that wait (`quieted`), and so do the slabs a disk read allocates on every device and the ones a copy beyond the tier frees, the one path the review's list of every such call found outside the rule's letter: creating and freeing a host-visible buffer is safe beside a recording on Vulkan, but the rule names one call, and the wait costs a read's start and end a stage's time at most, on a path that runs once a read, timed or not. Failing test first: `server-passes-cpu` runs a timed scheduler over two stages of CPU groups whose backends report a device time and count each reading that meets a submission of their own; 2 to 5 met at the test's commit in three runs, none at the fix.
- **Scope:** `--timing` with several passes in flight over groups, which no production server runs.
- **Left, parked (2026-10-07):** the review's list made a test. A first form, a test backend counting calls that meet a recording under a timed scheduler over two stages of CPU groups with a host and a disk tier, caught by chance: with each unguarded call planted in turn it failed every run for the timing read, one run in three for a promotion from host memory, never for a donor's copy to host memory, and its disk arm wrote and read nothing. The form to build is deterministic, as `server-resume`'s hook backend is for passes in flight: the backend holds a recorder inside a submission on a latch, the test causes each action from the scheduler's side (a donor's copy to host memory, its promotion, a disk read, the timing read), requires that no call from another thread reaches the backend while the latch is held and that it arrives once released, with the host tier sized from the scheduler's own figure for one copy and `disk_entries` and `disk_hits` required before that part runs. What blocks it: which call makes the host copies under that load is not found (the planted one changed no counter), nor why nothing went to disk. Until it lands, `server-passes-cpu`'s timed case holds the timing read alone and the other calls rest on the review's list in the collaboration log.
- **Reviewed by:** F2DEV at 7613411d, no blocking finding; the slab calls routed as it proposed.

## A short last message keeps its boundary too (2026-10-06, branch fix/boundary-last-block, lands by fast-forward)

- **Found:** in production (Qwen3.8-27B Q8_0 over two MI50s) a conversation's turn after a restart forked the boundary a message earlier, 74688 tokens, and not its previous prompt's state near 76608.
- **Cause:** a request keeps a message boundary where its last user message starts, but not where that block is also its own checkpoint, its prompt's last whole block, as it is for every message shorter than a block. The state then lived only in the request's donor, which the reply's job supersedes and which is therefore neither copied to host memory nor written at a stop.
- **Done:** the request keeps the boundary there too, one checkpoint being both (`Scheduler::admit`, `form`). The test commit fails without it: a request whose 50-token last message starts in its prompt's last whole block left no boundary.
- **Cost:** no checkpoint slot, and no copy where the turn before left its job's boundary at the same tokens, which is found and renewed: the usual client, now that a reply is read again as it is sent back. Where no such boundary exists, one state a turn is copied to host memory, 150 MiB on that model, in the room boundaries already live in, and with a disk tier written; a client that returns `reasoning_content` under the Qwen 3.8 template is that case on every turn.
- **Reviewed** by F2DEV.

## The held-write check makes its disk tier before its turns (2026-10-06, branch fix/held-write-ready, lands by fast-forward)

- **Found:** `server-resume`'s held-write case failed once in a hosted Windows job with "the cancelled write left 0 entries", on a branch that touches only the Vulkan backend.
- **Cause:** the check, not the server. On that runner the store was made only after the turns (its line follows the six requests in the log), so the turn that should have needed the room of a copy being written met no write, and the temporary file the check then waited for was the store's write probe, `probe.tmp`, or the first entry's write started after the turns, which nothing cancels. The check also listed the directory twice, once to wait and once to take the file.
- **Done:** the case waits for `Stats::disk_ready` before its turns, takes the file from the listing it waited on and requires it to be an entry's.
- **Shown under load:** twelve copies of the test at once on a 16-thread CPU: at main `85236d86` 2 of 24 runs fail with "a held write left in flight: the write's temporary file is gone, 0 entries kept", the text a hosted Ubuntu job gave that day, the same race in the case's four-copy form; with the change 60 of 60 pass. Reviewed by D2CDEV.

## The open int8 prompt cell is attention; the 8-bit tile shape is not built (2026-10-06, branch docs/int8-open-cell, docs only, lands by fast-forward)

- **Goal:** say why Qwen3-8B Q8_0 prompts at `--dtype int8` stay behind the reference fork on one MI50 after the two bit-identical levers (the two records of 2026-10-06 below on the attention tile and the block-major twin), and what would recover the cell.
- **Where a prompt's time goes**, device time of one prompt from `bench --profile` at c8b2ac8d, before those two levers:

  | prompt | total | the tile's matmuls (`matmul_tile_q8i8_tall`) | attention (`attention_tile_kv16_x8`) | norms, SiLU, rope and KV write |
  |---|---|---|---|---|
  | 2048 | 1804 ms | 1415 ms, 78.4% | 300 ms, 16.6% | 61 ms, 3.4% |
  | 4096 | 4192 ms | 2835 ms, 67.6% | 1175 ms, 28.0% | 126 ms, 3.0% |

- **Against the fork** at main 561d282a, both in one session on the same card: llmx reads 1169 and 1020 tok/s at pp2048 and pp4096, 1752 and 4016 ms a prompt, its attention 269 and 1050 ms measured; the fork reads 1257 and 1167 tok/s, 1629 and 3510 ms, its attention about 110 and 440 ms, an estimate from a fit over 512 to 4096 tokens.

  | prompt | gap to the fork | attention, llmx against the fork | everything but attention, llmx against the fork |
  |---|---|---|---|
  | 2048 | 123 ms | 269 against about 110, 159 ms behind | 1483 against about 1519, 36 ms ahead |
  | 4096 | 506 ms | 1050 against about 440, 610 ms behind | 2966 against about 3070, 104 ms ahead |

  Attention is behind by more than the whole gap at both lengths, and the rest of the prompt, the tile's matmuls included, is ahead by 2 to 3 percent.
- **Not built: the 8-bit tile's shape.** To close the cell alone the tile would have to lose 9 percent of its time at 2048 and 18 percent at 4096 while it already beats the fork's matmuls. On the 16-bit tile the pipelined step loop, one block a step, the 64-row tile everywhere and regrouped workgroups were each bit-identical and slower (the record of 2026-10-05 below on the 16-bit prompt tile), and the 8-bit tile's one layout lever, the block-major twin, gave 0.8 to 1.0 percent on this model.
- **Why attention cannot meet it inside the precision rule on this driver** (D2CDEV's finding, Mesa 25.0.7 read in source, agreed 2026-10-06): the fork's attention multiplies half pairs two an instruction, with an F32 sum in K.Q through `v_dot2_f32_f16` and a half-float sum over V. RADV emits no `v_dot2_f32_f16`, so the rule's form, 16-bit inputs with F32 sums, runs there at one product an instruction, which is today's kernel; and the half-float V sum is a lower precision than the rule allows.
- **The cell stays open**, recorded, not waived: pp512 level with the fork (1308 against 1313), pp2048 and pp4096 behind by 7 and 13 percent. The default attention keeps F32 sums and no lower-precision attention is built now.
- **Recovery, each on a later decision:** a backend that can emit the rule's two-product form, which the planned ROCm backend can, runs K.Q as the fork does inside the rule; a lower-precision attention on Vulkan would be an opt-in with a budget calibrated for that precision and frozen before any candidate runs, with the long-context checks, and whether it is a flag of its own or a wider meaning of `int8` is decided then, on the numbers. The cost probes of the 16-bit forms are in the collaboration log of 2026-10-05: about 1194 and 1067 tok/s with a half-float V sum, 1238 and 1137 with integer K.Q as well.

## The audit's int8 and drafter findings (2026-10-06, branch cleanup/audit-g, lands by fast-forward)

- **Goal:** D2CDEV's tightness audit of main since bd05390a, its findings on int8 and the drafters, each rechecked against the code.
- **Done:**

  | finding | recheck | disposition |
  |---|---|---|
  | `row_twin` and `tile_twin` are the same lines twice | true: both looked the tag up, quantized on a miss and bound the twin, differing in the quantizer, its group count and whether a block-major 8-bit twin is read | fixed: one `twin` holds the lookup and both quantizers; `row_twin` keeps its early return for float inputs and `tile_twin` the padded columns it reports |
  | bench's `--drafter` split into a bool and a string and rejoined twice | true | fixed: `bench_drafter` returns one `BenchDrafter` that `cmd_bench_model` takes |
  | `spec::pair` classifies a DFlash file that the loader then refuses, with a registry column for that alone | true: nothing runs a DFlash drafter until SPECULATIVE step 7 | fixed by removal: the DFlash checks, `DrafterKind::dflash`, the loader's refusal and `ArchEntry::dflash` are gone, and a DFlash file is refused by the check every unknown architecture meets ("its architecture 'dflash' is not one llmx runs"); step 7 brings its checks with their runner |
  | the 8-bit producers in `xquant.glsl` parallel the 16-bit ones | true, about ten lines shared | rejected for its value: a shared core saves about ten lines; taken up if those functions are next edited for a reason of their own |

- **Result:** src/ loses 23 lines net and tests/ 18; no behaviour changes but the text of the DFlash refusal. The bytes of long-prompt logits, perplexity and decode rows under int8 and f16 on Qwen3-8B Q8_0 and Q4_K_M equal main's on one MI50.
- **Review:** F2DEV read the branch at 702dfeda5 and posted no finding; it agreed with leaving the `xquant.glsl` core for its value.

## A chat reply is read again as its client sends it back (2026-10-06, branch fix/follow-content, lands by fast-forward)

- **Found:** in production (Qwen3.8-27B Q8_0 over two MI50s, Open WebUI on `/v1/chat/completions`) the turn after a long reply waited minutes for its first token, restart or not: after "74766 prompt tokens, 6247 generated" the next turn was "76655 prompt tokens (74752 reused)", and the 123200 rows the server had read again while idle were never forked.
- **Cause:** the re-read rendered the reply as the route gave it, its reasoning beside its content (`reasoning_content`). The Qwen 3.8 template keeps the reasoning it is given in every assistant turn, where the older templates drop a past turn's, and clients send back the content alone, which the template renders with an empty think block. So the history read again and the real next prompt parted at the reply's first token.
- **Done:** `reply_sent_back` (`src/server/api.hpp`) is the assistant turn a reply is read again as: on the compatible route the content alone, read as the route reads a message without `reasoning_content`. The test commit fails without it, in `server-utf8`: under a template that keeps the reasoning it is given, the turn held the reasoning and rendered another prompt than the content sent back does.
- **Measured** on two MI50s, Qwen3.8-27B Q8_0 with the embedded drafter, an 825-token prompt whose reply is 9588 tokens (24811 characters of reasoning, 1543 of content), the follow-up sent with the content alone, 1212 prompt tokens:

  | | rows read again | follow-up reused (tokens) | follow-up first token (s) |
  |---|---|---|---|
  | main `1a0e92d8` | 10368 | 768 | 1.36 |
  | this branch | 1152 | 1152 | 0.33 |

- **A client that does return `reasoning_content`**, under a template that keeps it, now reuses the conversation to its previous prompt's last whole block and reads that reply again, as every client did before; one re-read serves one of the two renders, and the standard request carries no reasoning.
- **Reviewed** by F2DEV.
- **Left:** a request whose last user message starts in its prompt's last whole block keeps no message boundary of its own, so after a restart its conversation is forked a message earlier; its own branch follows.


## The known findings cleaned up (2026-10-06, branch cleanup/known-findings, lands by fast-forward)

- **Done:** `tests/data/known_findings.txt` goes from 34 lines to 15, none waiting for a branch. Removed as unreached: `Backend::state_copy` (its reason, step 8b of the qwen35 plan, was false: nothing but two tests called it), the `Fd(int)` constructor of the hub transport, the const `kh` and `vh` of the CPU KV storage and `BlockPool`'s sizing constructor; `http::fetch`, the HTTP layer's client, moved into the `http` test, its one user. The 13 line numbers STATUS's records gave without a commit are dropped, `docs/VULKAN.md` no longer writes a command that does not exist, and `q8_dots.hpp` has its page.
- **Kept, with reasons that hold:** the Windows entry point of `llmx-model-logits`; the four probes the backend tests read (`workers_started`, the Vulkan device's name, profile and kernel representations) and what they return; the const reads of a KV block and a sequence's block count and move assignment, which `kv-cache` takes.

## A message boundary where a request's last user message starts (2026-10-06, branch feat/edit-boundaries, lands by fast-forward)

- **Done:** a chat request whose prompt passes the start of its last user message, having forked below it, keeps a message boundary at the whole block below that start, which the next edit of that message forks (`Api::message_start_of`, one rule for both chat routes); with a disk tier a conversation's boundaries are not thinned; and host memory takes the entry whose write to disk is in flight last (`Scheduler::written_soon`), since taking it first cancelled every write once the writer had fallen behind, which the added boundaries brought about at 24 users (29 copies dropped unwritten, none with the rule).
- **Measured** (Qwen3.8-27B Q8_0 on one MI50, default clocks, `--max-seqs 8 --ctx-size 32768`, 64 GiB of disk tier; main `72309913`, this change and the reference server in one session a workload; twenty turns a user, then a regenerate and an edit of turn 2; time to first token in seconds, p50/p99):

  | | follow-up | regenerate | edit |
  |---|---:|---:|---:|
  | 6 users, main | 2.30/3.57 | 2.84/4.08 | 3.19/4.10 |
  | 6 users, this change | 2.29/3.66 | 2.06/2.38 | 2.13/2.94 |
  | 6 users, reference server | 2.58/3.31 | 4.98/6.01 | 1.55/1.84 |
  | 24 users' 14 long conversations, main | 2.31/23.63 | 5.03/5.59 | 4.64/5.62 |
  | 24 users' 14 long conversations, this change | 2.67/4.35 | 5.29/6.01 | 1.84/2.89 |
  | 24 users' 14 long conversations, reference server | 15.52/29.52 | 4.98/5.52 | 1.57/1.86 |

  No follow-up reused less than main's. At 24 users six of main's long conversations still had the boundary before message 2 on disk in this run and regenerated in under 3 s, none in three earlier runs and none of this change's: the boundaries alone are 72 GB there against the 64 GiB tier, the oldest go first, and this change's fill it sooner.
- **Left:** the rest of the edit gap is the boundary's whole block (25 to 37 tokens), the prompt rate (4.3 ms a token against the reference's 3.9) and the regenerate's job pass an edit waits behind; a floor on a kept boundary's position, if regenerates at 24 users are to be level, is its own measured change.
- Reviewed by D2CDEV.

## A decode sum at width 4: the reference's row and where the time goes (2026-10-06, measured, docs only, lands by fast-forward)

- **Same shape or not (2026-10-07):** the width 4 cells (llmx's one group of four against the reference's tensor split of four) are same-shape; the two-stage and layer-split columns, where they stand beside the reference, are not. The same-shape table is the record "Llmx against the reference in the same topology", above.

- **Goal:** the tensor split's open decode cells at width 4, measured before anything is built: the reference's tensor split of four beside llmx on the same cards, and what a decode sum's time is made of at width 4 against width 2.
- **Measured** (main a01216fb, `llmx 0.1.0+ga01216fb202f`; Qwen3-32B Q8_0 on four MI50s under one root, default clocks, the process on logical CPUs 8 to 11, one session, arms interleaved, two rounds; `bench --model --p 512 --n 128 --r 3`, and the reference `llama-bench -ngl 99 -fa on -sm tensor -lm dio -p 512 -n 128 -r 3` of the pinned image with its environment; tok/s):

  | | width 4 pp512 | width 4 tg128 | width 2 pp512 | width 2 tg128 |
  |---|---|---|---|---|
  | reference tensor split | 502.1, 511.1 | 49.8, 48.9 | 569.5, 568.9 | 33.3, 33.9 |
  | llmx `--dtype int8` | 588.3, 586.2 | 12.1, 12.2 | 532.6, 532.4 | 26.6, 29.6 |
  | llmx default precision | 460.1, 459.3 | 12.3, 16.5 | 356.6, 357.6 | 27.6, 29.7 |

  At width 4 and `int8` a prompt reads 1.16 times as fast as the reference's and decode runs at a quarter to a third of it. The width-4 decode cell read 15.6 and 16.3 an hour earlier in the same session.
- **Where a decode sum goes** (`int8`, timing-only builds of the same commit, their logits wrong where the sum is cut; tg128, two rounds):

  | | width 4 | width 2 |
  |---|---|---|
  | main | 15.6, 16.3 | 29.7, 29.5 |
  | no sum at all | 61.6, 62.1 | 36.0, 36.3 |
  | semaphores and adds, nothing copied to the peers | 16.3, 15.1 | 30.5, 29.6 |
  | a token's time over its 128 sums, and with no sum | about 470 and 126 us | about 262 and 217 us |
  | so a sum costs | about 345 us | about 46 us |
  | host time in a one-row sum: recording the copies | 11 us | 2 us |
  | the members' submissions, one after another, wall and the thread's CPU time | 290 and 284 us (four) | 206 and 117 us (two) |
  | exporting and importing the sync files, wall and CPU | 107 and 107 us (twelve) | 19 and 19 us (two) |
  | recording the adds | 26 to 29 us | 10 to 15 us |

  The copies are not the cost at decode. At width 4 a sum is host CPU time on the one thread that records, about 390 us of it in the members' submissions and their sync files, and that thread paces the token while the devices have 126 us of work an interval. At width 2 the devices have 217 us of work an interval, the host is mostly ahead, and the sum costs 46 us. A submission that waits on and signals sync-file semaphores takes 58 to 71 us of CPU here against about 7 us for a plain one in `llmx-vk-handoff exchange` (its floor).
- **What llmx could cut without the kernel route** (estimates from these numbers, none built): each member's submission and sync-file work on a thread of that member's, up to three quarters of the 391 us at width 4, about 43 tok/s as an upper bound, and bounded by the devices at about 33 to 36 at width 2; one sync file a member in place of one a pair, about half of the 107 us; whatever makes a semaphore submission ten times dearer than a plain one, unknown until profiled; sums fused into the kernels that produce and consume them, about 40 us a sum. A ring or tree at decode sizes adds rounds, and submissions are the cost; one submission for a layer's two sums is not possible, since the feed-forward reads the first sum.
  Only a wait inside one submission (`docs/TENSOR-SPLIT.md`, the kernel route) or another transport removes the submission a sum; decode would then approach the figures with no sum, 62 tok/s at width 4 and 36 at width 2.
- **Gotcha:** this machine's logical CPUs 8 to 11 are the second threads of cores 0 to 3, which other work on the machine uses, so a path bound by one host thread, as width 4's decode is, moves with what runs there (12 to 16.5 tok/s in this session).
- **The profile of the exchange** (strace of every ioctl and 120 stacks of the recording thread of a decoding process; the machine has no kernel profiler, so the kernel's side is seen by call): the thread is inside an ioctl in four stacks of five at both widths, and one sum at width 4 is about 76 calls, 4 submissions, 12 syncobj waits and five calls for each of the twelve sync files exported and imported, against about 16 at width 2. No single call is dear; the count grows with the pairs of members. Why a submission's own ioctl takes about 30 us here against about 7 in the exchange probe is not answered.
- **The levers timed** (timing builds of the same commit with only the exchange rebuilt, greedy ids those of main at both widths; `int8`, two rounds, the sibling threads of the process's CPUs 2 to 8 percent busy):

  | tg128 | width 4 | width 2 |
  |---|---|---|
  | reference tensor split | 48.9, 49.8 | 33.3, 33.9 |
  | main | 16.5, 16.6 | 29.7, 29.8 |
  | one sync file a member, merged by the waiter | 17.8, 17.8 | 29.6, 29.7 |
  | a thread a member for its submission and sync files | 23.7, 25.2 | 29.6, 29.5 |
  | both | 22.7, 21.8 | 29.7, 29.7 |

  pp512 does not move with any of them. A thread a member gains about 1.5 times at width 4, not the three quarters of the host time estimated above: the build wakes its threads three times an exchange, and not all of that time divides. Without the kernel route the sync-file exchange at width 4 reaches about half the reference's decode in the best of these builds, and width 2 has nothing left on the host's side.
- **Left:** the decision what to build; none of the timing builds is in the tree.

## The tensor split's probes and duplicates cut (2026-10-06, branch cleanup/tp-audit, the tightness audit's shares C and F, lands by fast-forward)

- **Goal:** D2CDEV's audit of main since bd05390a, shares C and F: tool modes that answered their question, and small duplicates in the tensor split's code.
- **Result:** `llmx-multi-device-bench` keeps `stages`, the stage-time source of the split models, and loses `concurrent`, `pipeline` with its simulated drivers, `groupsum` and `exchange`; `llmx-vk-handoff` keeps `probe` and `exchange` with device inboxes, the two a new driver is held to (`docs/TENSOR-SPLIT.md`, step 0), and loses `time`, `pingpong` and the host-inbox variant with its relayed chains. Their measurements stay in the records below and in `docs/MULTI-DEVICE.md`. The CLI no longer makes and drops a collective to ask whether a group has a sum, which the model refuses as it is made, so that refusal now comes once the file's headers are read and after the plan's refusals, no longer before the file is opened; the KV heads a member keeps are one function (`kv_share`); the head step's unread width is gone. Two records are corrected: a bench mode named that never landed, and `stages` said to read layer counts from the loaded model, which it parses from the plan's text.
- **Kept, with the reason:** the CLI's checks of a width (whole groups, one share a group, one kind), which are usage errors refused before a model file is read with status 2, as `docs/USAGE.md` and `docs/TENSOR-SPLIT.md` section 4.6 state and `tests/cli.py` holds, while the same conditions in `place_model` and the model are those layers' own invariants for callers that are not the CLI; `footprint`'s member, through which `tests/shard.cpp` holds every member's footprint to its shards, the equality the fit's use of member 0 rests on; `llmx-vk-handoff probe`, the only listing of a new driver's handle types; and the linear attention's shards, which the planner reaches and the model refuses above width 1 until step 5 of TENSOR-SPLIT.
- **Reviewed by:** S2DEV at d30f51c8: lands after its five points, each taken in this commit; its question on a model of no KV heads is answered by the file's reader, which refuses a count that is not positive (`metadata::integer`) before either architecture's configuration divides by it.

## A group's large sums in two shots (2026-10-06, branch perf/tp-twoshot, step 3b of TENSOR-SPLIT, lands by fast-forward)

- **Goal:** the user's direction of 2026-10-05, a collective that chooses by a sum's size: among three members or more, broadcast sends every member each whole partial, which is what a prompt pass at width 4 spent its time moving.
- **Result:** inside the Vulkan backend's collective a sum of 327680 floats or more (1.25 MB) among three members or more sends member k only share k of each partial, which k sums in member order and sends on; smaller sums, a decode step's among them, and every sum between two members stay broadcast. Every float is the same sum in the same order either way. The fit counts a member's two gather rows. Nothing outside `vulkan_backend.cpp` knows the way a sum takes.
- **Measured** (95019267, `llmx 0.1.0+g95019267aa15`, against main bdd61396, both built the same way from detached worktrees; Qwen3-32B Q8_0 on four MI50s under one root at `--tensor-width 4`, default clocks, `bench --model --p 512 --n 128 --r 3`, the arms interleaved main, branch, branch, main twice):

  | tok/s, four runs each | main | two shots |
  |---|---|---|
  | pp512 | 318.9 to 319.6 | 458.6 to 460.7 |
  | tg128 | 16.5 to 17.2 | 16.5 to 16.9 |
  | pp512, `--dtype int8` | 370.0 to 371.0 | 585.4 to 587.5 |
  | tg128, `--dtype int8` | 16.3 to 16.9 | 16.2 to 16.9 |

  Greedy ids and the last logits row of the excerpt are the same bytes on both arms, the check that reaches two shots in a model: the tiny fixtures' sums stay below the crossover and are broadcast.
  Checks at that head: CTest 45 of 45 with four MI50s, `backend-vulkan` summing among three on both sides of the crossover; docs, dead-code and the linked check; the suite's `tensor-split` and `chat` at width 4.
  The crossover comes from the exchange alone, timed on the same cards before this change: two shots never win between two members, are level with broadcast at 640 KB and ahead from 1.25 MB among three and four (10 MB: 3.2 against 4.7 ms among three, 3.8 against 7.6 ms among four).
  Reading every peer's partial in place, one shot, was slower than both at every size and is not in the code.
- **Left:** the decode sum at width 4, which two shots do not touch (measured in the block above).
- **Reviewed by:** D2CDEV at 95019267, no finding; its note on the machine that holds the two-shot path is in AGENTS, and the crossover stays a constant of the Vulkan backend until a second transport or card gives a second number.

## The kept-entries check waits for its reads (2026-10-06, branch fix/keep-check-wait, lands by fast-forward)

- **Found:** with its first turns past 449 tokens (the `fix/keep-check-split` block, below), the kept-entries check still failed on one MI50 about every second run: the third conversation's follow-up reused 0 tokens, 2 of 4 runs of the scenario at main `6ea7ae5d`, with all three entries adopted and all three read (`disk_hits` 3, no error).
- **Cause:** the check again, not the server. A request waits for its read at most as long as computing the shared tokens would take at the measured prompt rate; on an MI50 Qwen3-0.6B Q8_0 reads 448 tokens in less time than a 133 MB entry just adopted takes to read, so the request is admitted without it and the read completes behind it. The check asserted every follow-up reuses its history, which the server only promises while no rate is measured.
- **Done:** the check's servers read prompts in passes of 16 rows (`--ubatch 16`), which measure no rate, as `server-resume`'s read-back checks do; 8 of 8 runs of the scenario on the MI50 then reuse 512, 512 and 448 tokens.
- **Correction to that block:** its MI50 run of the `server` component passed once and was taken as the fix holding; one run could not show a failure that comes every second time.

## A group's stage recorded off the scheduler's thread (2026-10-06, branch feat/tp-stage-threads, a step of TENSOR-SPLIT after step 4, lands by fast-forward)

- **Same shape or not (2026-10-07):** the reference columns are its tensor split of four (one group), beside llmx's two stages of two and its layer split of four, so "leads the reference from 16 users" and the cells open against the reference compare different shapes; none of the record's reference cells is same-shape. The same-shape table is the record "Llmx against the reference in the same topology", above.

- **Goal:** stages of tensor groups overlap. Recording a group's stage runs in step with its devices, so the scheduler's one thread was held for the whole stage and two stages of two served no faster than one group (tensor groups serving, below).
- **Result:** with several passes in flight each group's stage has a recorder thread that records one stage of one pass at a time and decides nothing (`docs/SERVER.md`, The round); the round and its policy stay the scheduler's, `round_steps` taking the fact that a pass is being recorded. `backends/backend.hpp` states the one call another thread may make on a recording backend, `wait` of a returned ticket. A layer split's stages are recorded inline as before.
- **Measured** (0f02aa41, `llmx 0.1.0+g0f02aa41fe37`, binary sha256 00827e876db102c7; Qwen3-32B Q8_0 on MI50s at default clocks, `serve --max-seqs 64` with a pool of 81920 tokens on every arm, `tools/server_load.py` with 128 generated tokens at 1 / 4 / 16 / 32 / 64 users, greedy, the arms in turn from fresh servers in one session a table, the reference `llama-server -sm tensor` of the pinned image on the same cards; no request failed; the machine's load average 5 to 29 with other work on other cards and cores). Output tok/s, then the inter-token p99 in ms:

  | four MI50s under one root | two stages of two, `--dtype int8` | two stages of two | layer split of four | reference tensor split of four |
  |---|---|---|---|---|
  | 128-token prompts | 26.5 / 90.7 / 230.7 / 288.3 / 266.1 | 25.7 / 79.4 / 172.8 / 180.5 / 184.6 | 15.8 / 57.0 / 155.1 / 192.7 / 177.6 | 43.5 / 105.2 / 143.9 / 200.3 / 189.3 |
  | 1024-token prompts | 17.9 / 51.0 / 81.7 / 87.6 / 84.1 | 15.5 / 40.0 / 57.9 / 59.5 / 60.1 | 10.2 / 35.2 / 60.9 / 65.4 / 64.6 | 27.5 / 42.8 / 46.2 / 49.2 / 49.7 |
  | prompts of 64 to 1024 | 19.0 / 55.0 / 117.3 / 124.9 / 122.0 | 16.4 / 43.6 / 87.6 / 81.9 / 80.6 | 10.6 / 36.2 / 79.4 / 85.9 / 86.7 | 29.4 / 47.9 / 70.2 / 75.9 / 76.1 |
  | p99, 128-token prompts | 37 / 39 / 55 / 825 / 1203 | 40 / 46 / 71 / 1374 / 1683 | 61 / 56 / 76 / 136 / 2578 | 42 / 80 / 119 / 143 / 4093 |
  | p99, 1024-token prompts | 42 / 529 / 1128 / 1168 / 1243 | 42 / 742 / 1598 / 1691 / 1791 | 70 / 60 / 2752 / 2861 / 3010 | 49 / 74 / 4228 / 4393 / 4396 |
  | p99, prompts of 64 to 1024 | 45 / 208 / 1204 / 1178 / 1323 | 39 / 222 / 1677 / 1856 / 2034 | 70 / 65 / 1799 / 3006 / 3347 | 47 / 72 / 4406 / 4515 / 4553 |

  | eight MI50s | two stages of four, `--dtype int8` | two stages of four | layer split of eight | reference tensor split of eight |
  |---|---|---|---|---|
  | 128-token prompts | 12.7 / 35.4 / 125.8 / 212.7 / 242.9 | 14.8 / 50.3 / 163.8 / 211.7 / 219.2 | 13.4 / 57.7 / 163.6 / 277.2 / 315.7 | 43.3 / 89.9 / 116.7 / 145.0 / 155.7 |
  | 1024-token prompts | 12.1 / 33.0 / 63.1 / 71.7 / 71.2 | 9.8 / 29.0 / 48.1 / 49.2 / 55.2 | 8.8 / 36.3 / 80.7 / 113.3 / 111.1 | 26.7 / 36.5 / 37.4 / 30.5 / 31.5 |
  | prompts of 64 to 1024 | 12.5 / 35.8 / 92.8 / 107.0 / 106.9 | 11.7 / 33.8 / 84.9 / 85.5 / 90.2 | 9.1 / 40.4 / 99.6 / 137.1 / 155.8 | 28.1 / 41.3 / 59.1 / 60.2 / 60.1 |
  | p99, 128-token prompts | 99 / 193 / 204 / 1207 / 1604 | 69 / 74 / 73 / 1419 / 1757 | 72 / 58 / 73 / 76 / 1808 | 79 / 183 / 156 / 313 / 5182 |
  | p99, 1024-token prompts | 63 / 707 / 1483 / 1511 / 1584 | 299 / 840 / 1806 / 2123 / 2497 | 73 / 61 / 1725 / 2767 / 2827 | 59 / 140 / 5413 / 7475 / 6828 |
  | p99, prompts of 64 to 1024 | 67 / 497 / 1469 / 1522 / 1590 | 71 / 643 / 1734 / 1798 / 1918 | 73 / 61 / 453 / 2714 / 3381 | 61 / 134 / 2486 / 5595 / 5578 |

  Before the threads two stages of two gave 26.0 / 44.7 / 95.0 / 96.5 / 112.0 on 128-token prompts (step 4's table).
  On four cards at `int8`, the gate's arm, the staged tensor split leads the reference's tensor split from 16 users on under every load and at 4 users under the two longer ones.
  On eight cards another session's identity and timing runs used one of the cards from 06:03 to 06:31 UTC, beside every 128-token arm but the first and every 1024-token arm, so those cells are kept and may read low.
  On eight cards the second group spans three roots; a stage of four is slower than a stage of two, its sums costing more than its members save, and the layer split of eight is llmx's fastest arrangement there from 4 users on.
- **Open against the reference at the same precision** (first support of stages of groups): on four cards one user under every load (26.5 against 43.5, 17.9 against 27.5, 19.0 against 29.4), 4 users on 128-token prompts (90.7 against 105.2), and the inter-token p99 at 4 users under the two longer loads and at 32 users on 128-token prompts (825 against 143 ms); on eight cards 1 and 4 users under every load (4 users on 128-token prompts 35.4 against 89.9) and the p99 at 4 and 16 users on 128-token prompts and at 4 users under the longer loads. At default precision the staged split is also behind at 32 and 64 users on 128-token prompts on four cards.
  The recovery: the sums' cost (step 4's open cells: width 4's host time and two shots), and for the p99 a cap on the prompt rows of a pass that carries decoders. Every gap above 0.4 s falls where a decoder's row rides a pass carrying a whole ubatch of prompt rows; with `--ubatch 128` the four-card `int8` arm gives 26.7 / 90.7 / 226.4 / 279.5 / 260.9 tok/s at a p99 of 42 / 44 / 323 / 340 / 616 ms, half the p99 at 32 and 64 users for 3 percent of the output and a worse one at 16.
- **Found by the two-stage load check and fixed before landing:** at 8985e38a the server stopped under mixed prompts at 64 users. The scheduler's thread waited for idle recorders while it held its own lock, reached from admission dropping a donor to host memory, and both recorders waited for that lock to report a recorded stage, one of them already handed its next pass. A recorder now reports that it is idle under its own lock and its result under the scheduler's, a pass is handed over again only once the result is out, and a result names its hand-over so one arriving after its pass was abandoned is dropped. No CTest reaches it: it takes two stages recording on devices while admission drops a donor, and the simulation is one thread with no locks. The hand check that holds it is this load: two stages of two on four MI50s, Qwen3-32B Q8_0 at `--dtype int8`, mixed prompts of 64 to 1024 tokens at 16 to 64 users, at every change to the recorders.
  At the fixed head (b438b74e, GPU[2]+GPU[3] and GPU[4]+GPU[6], the second group across two roots, one session with the head measured above): four mixed loads 18.9 to 19.1 / 52.8 to 56.7 / 121.2 to 121.8 / 124.4 to 125.6 / 116.7 to 130.2 tok/s against 19.2 / 55.5 / 123.3 / 127.4 / 120.5 before the fixes, 128-token prompts 26.6 / 89.5 / 231.1 / 291.8 / 278.0 against 26.5 / 88.7 / 230.3 / 292.5 / 317.3, and 1024-token prompts 18.2 / 50.6 / 83.0 / 89.9 / 89.1; no request failed and no server hung. The 64-user cell on 128-token prompts read 266 and 317 for the earlier head in two sessions, so 278 is inside what one run of that cell gives.
- **Also from review:** every abandon of one pass waits for idle recorders first, since abandoning drains every device; a round that could do nothing while a stage records waits, without polling, for a recorder, an arrival, the disk tier or its next bound, and a cancellation is seen at the next round, within a stage's time; while admission drops a donor beside a recording stage, a new submission and `/v1/health` are held for up to that stage's time.
- **Checks** (b438b74e, the code that lands, on main 737b5fa4; then rebased onto 1a0e92d8 with no conflict): CTest 45 of 45 on two MI50s; `server-passes` 2000 schedules with 116765 stages recorded on threads of their own, 2425 failing as they ended; `server-passes-cpu` and `server-resume` with no ThreadSanitizer report; docs, dead-code and the linked check; the suite's `server` and `chat` on one MI50. At 0f02aa41, before the review's changes to the scheduler: byte identity of Qwen3-0.6B, Qwen3-8B and Qwen3.5-0.8B Q8_0 at width 1 on the CPU and on one MI50 in each load mode, the suite on the CPU and on one MI50, Qwen3-8B Q8_0 on one MI50 pp512 838 to 840 against 837 to 841 tok/s and tg128 77.1 to 77.2 against 77.1 to 77.5, and `tools/server_mix_check.py` on Qwen3-8B Q8_0 over two stages of two, each reply its reply alone.
- **Reviewed by:** D2CDEV at 1dd3cfff (the design as built, one finding and one question, both fixed); F2DEV at 8985e38a and b438b74e (the wait for idle recorders before an abandon, the exact wait, and the deadlock's fix; no finding, its notes taken or answered in the collaboration log).

## Tensor groups serving (2026-10-04, branch feat/tp-staged-serve, step 4 of TENSOR-SPLIT, lands by fast-forward)

- **Same shape or not (2026-10-07):** the width 2 cells (a group of two against the reference's tensor split of two) are same-shape; the cells of four and eight cards (llmx's two stages of two or four, or its layer split, against the reference's one tensor split of four or eight) are not. The same-shape table is the record "Llmx against the reference in the same topology", above.

- **Done:** the cause of the server cells, measured on Qwen3-32B Q8_0 over GPU[2] and GPU[3] (main's code, 9e687440 and 235375a9, RADV, default clocks); it is not in how passes are formed, so it goes to step 7 with these numbers:
  - at 32 users of 512/128 both arms read prompts at about the same rate, the group 215 and the layer split 225 tokens a second; served with `--timing`, the group's one stage carried 299 rows a second of device time against the layer split's busiest stage's 280, idle 13.7 percent against 8.8 and 4.2, so the group's device work is no slower and its loss is the time its stage stands idle;
  - two passes in flight on the group's one stage, tried by lifting the one-pass rule in the scheduler and in `reserve_passes` (replies still each its reply alone in `tools/server_mix_check.py` on Qwen3-8B Q8_0), gave 50.2 tok/s against 53.4 with one, idle 14.1 percent, so the idle is not the host's gap between passes; not kept;
  - the sums' share, from builds that leave part of the sum out (timing only, their logits wrong), `bench --p 512 --n 128 --r 2`, interleaved twice: the group pp512 357.9 / 356.9 and tg128 29.76 / 29.81 tok/s; without any sum 401.1 / 399.5 and 36.45 / 36.40; with the semaphores and adds but no copy across the cards 395.4 / 394.7 and 30.42 / 30.29; so a prompt pass loses about 10 percent to the copies of the partial rows across the cards, about 1.1 ms for each 10.5 MB copy at 512 rows, 128 a pass, and decode loses about 17 percent to the waits, about 48 us a sum; copies and adds with the semaphores left out ran 179.7 and 12.3, the members then waiting on each other's whole submissions through the dma-buf's implicit sync;
  - so the copies are what step 7's overlap hides (a micro-batch exchanging while the other computes), the waits what the kernel route of section 8 would shorten, and the gap to the reference's prompt speed is mostly one card's: the group without any sum reads about 200 tokens a second a card against the reference's 285 at width 2.
  The fit counts what a group's member keeps beside its layers: its collective's rows, which each backend reports (`Backend::collective_bytes_per_row`: the CPU its partial row, Vulkan its partial and scratch rows and two parities of an inbox of a slot a member, about 47 MB a member on Qwen3-32B at width 2 and 576 rows, four rows' worth of 5120 floats a row), and on the head's group its slice of every logits row, the host's logits rows now counted at the whole vocabulary rather than a member's slice (`placement`).
  A group may span PCI root complexes (the user's question, 2026-10-05): `llmx-vk-handoff exchange` with dma-buf inboxes, 64 epochs, median of 5 chains, three runs, every sum correct, took 119 to 130, 145 to 158, 369 to 383 and 2153 to 2213 us an epoch at 20 KB, 160 KB, 1.25 MB and 10 MB across roots (GPU[5] with GPU[6] or GPU[7], lent) against 121 to 127, 147 to 155, 376 to 385 and 2177 to 2192 on one (GPU[6] and GPU[7]), since the collective only writes into its peers' memory; `join` now refuses only a group whose memory or semaphores cannot be shared, naming each device and its root (`vulkan-lifetime`), and the command line prints each group's roots at startup and the `--device` order that puts every group under one root where the list allows it (`infer::one_root_order`, printed by `report_groups`; `Backend::pci_root`).
  The server gate on Qwen3-32B Q8_0 (72cf0670, `llmx 0.1.0+g72cf067035ee`, binary sha256 2ae10e581491d489; the reference image `mxxm/mx-llama.cpp:gfx906` at eefc4e732, `-sm tensor -ngl 99 -fa 1 -lm dio -np 64 -cram 0`), `serve --max-seqs 64` with a pool of 81920 tokens on every arm, `tools/server_load.py` at 1 / 4 / 16 / 32 / 64 users, 128-token replies, each arm from a fresh server and the arm order rotated a load, RADV, default clocks, output tok/s, greedy (the defaults within 3 percent of it in every cell):

  | two MI50s (GPU[2], GPU[3]) | group of two | layer split | reference tensor split |
  |---|---|---|---|
  | 128/128 | 27.4 / 74.3 / 95.0 / 110.5 / 108.2 | 17.6 / 49.3 / 101.4 / 106.1 / 116.3 | 29.9 / 76.8 / 106.7 / 165.6 / 192.1 |
  | 1024/128 | 17.0 / 28.3 / 31.1 / 31.8 / 30.6 | 12.0 / 24.5 / 34.0 / 34.2 / 34.8 | 21.4 / 38.0 / 42.6 / 48.0 / 50.0 |
  | mixed, prompts of 64 to 1024 | 18.4 / 32.4 / 46.9 / 45.5 / 45.3 | 12.5 / 26.5 / 47.9 / 45.2 / 47.0 | 23.3 / 41.8 / 60.5 / 70.0 / 75.0 |

  | four MI50s (GPU[4], GPU[5] and GPU[6], GPU[7], lent) | two stages of two | layer split of four | reference tensor split of four |
  |---|---|---|---|
  | 128/128 | 26.0 / 44.7 / 95.0 / 96.5 / 112.0 | 15.4 / 55.4 / 151.4 / 193.4 / 192.3 | 43.6 / 105.6 / 139.8 / 197.7 / 183.4 |
  | 1024/128 | 14.6 / 20.9 / 30.8 / 31.2 / 32.2 | 10.0 / 33.9 / 59.8 / 66.1 / 61.2 | 27.6 / 42.8 / 45.7 / 49.3 / 45.0 |
  | mixed | 16.6 / 25.4 / 46.9 / 44.4 / 46.8 | 10.7 / 38.0 / 80.8 / 85.5 / 86.0 | 29.4 / 47.8 / 70.0 / 72.0 / 76.6 |

  Inter-token p99 at 32 users of 128/128: the group of two 1626 ms, the layer split 2372, the reference 178; on four cards the two stages 1287, the layer split 141, the reference 161. No request failed on an llmx arm; the reference's server ended during its 128/128 run at the defaults on four cards (32 users, all 32 failed), its greedy run of the same load whole.
  So one request is faster on a group than on a layer split (27 against 18 tok/s) and the group leads to 4 users; from 16 users the group of two is level with the layer split or below it by up to 10 percent, and the reference's tensor split leads every cell. Two stages of two serve no faster than one group of two and half the layer split of four: the stages do not overlap, since a group's stage is recorded as two submissions a layer and the host waits in them for the device, the command ring holding 16, where a layer split's stage is one submission; that is this step's open defect.
  Measured (four MI50s, 32 users of 128/128, `serve --timing`): the two stages stand idle 57 and 56 percent of the time, the layer split's four 29 to 35, at the same device rows a second (386 against 398), and a round's host recording takes 93 ms against the layer split's 3.3.
  With a ring of 256 command buffers and no hold at a sum's exchange (not kept) the host still waits: a submission then takes 222 us on average in the driver, which lets a context keep 32 jobs in flight (`amdgpu.sched_jobs`), fewer than the 64 sums a member takes a stage, so the recording thread runs in step with the device whatever the ring holds; serving rose from 96.5 to 121 tok/s at 32 users, the stages idle 48 and 46 percent.
  Not the cause, each tried in a timing-only mode of `llmx-multi-device-bench` that never landed: the device's buffer count (3000 small or 700 of 32 MB change a sum by under 5 us) and host memory imported into the members.
  So a group's stage has to be recorded off the scheduler's thread, a thread a stage as the plan's risk 2 foresaw, for stages of groups to overlap; that is a change to the pass engine and the scheduler's round, proposed as its own step.
  The `server` and `chat` components run under a tensor width on a model of the tensor-split fixtures' shape: the server's first check, held to the CLI on the same group, and chat's conversations, the same at every thread count and ubatch, with its other checks.
- **Left:** the server load on eight MI50s as two stages of four with the skew check there, and width 8 measured once before the cap of 4 is decided, wait for the stage threads, without which stages of groups do not overlap.
  A group of three or more ran its members in turn (found 2026-10-05): Qwen3-32B Q8_0 at width 4 under one root (GPU[2] to GPU[5]) gave pp512 82.9 and tg128 7.9 tok/s, against 751 and 62.9 with no sum at all, where width 2 loses a fifth. Each member's inbox was one dma-buf every peer imported, and the kernel orders an importer's submission behind the other importers' submissions that listed it, an exporter's own being exempt, so two members never met it and three did. Each inbox now has one importer, a buffer for each peer, which also drops the slot a member kept for itself (d8824ff0; a prompt's logits the same bytes before and after at widths 2 and 4): width 4 then gives pp512 318 and tg128 16.4 tok/s (369.6 and 16.4 at `--dtype int8`), width 2 unchanged at 357.6 and 29.5 (526 and 29.6 at `int8`), two runs each.
  Width 4 is still below width 2: with no sum it would run 751 and 62.9 tok/s, so a decode sum costs it about 350 us and a 512-row sum about 7 ms; of the decode sum most is the host's, four submissions and twelve sync files exported and imported a sum, which is the next thing measured, and of the prompt's sum the copies, three partials of 10 MB a member, which two shots halve.
- **The gate's arm, `--dtype int8`,** on the two MI50s (d8824ff0, `llmx 0.1.0+gd8824ff03` and its hash, the same load and pool, greedy, output tok/s at 1 / 4 / 16 / 32 / 64 users, the group at the default precision run again beside it):

  | two MI50s | group of two, `int8` | group of two, default | reference tensor split (above) |
  |---|---|---|---|
  | 128/128 | 27.4 / 82.3 / 151.0 / 165.8 / 150.9 | 25.7 / 71.0 / 93.7 / 108.3 / 102.4 | 29.9 / 76.8 / 106.7 / 165.6 / 192.1 |
  | 1024/128 | 19.0 / 36.3 / 45.0 / 45.8 / 44.6 | 16.5 / 27.6 / 30.5 / 31.9 / 30.3 | 21.4 / 38.0 / 42.6 / 48.0 / 50.0 |
  | mixed | 20.2 / 40.9 / 69.7 / 67.2 / 64.5 | 17.8 / 32.0 / 46.4 / 45.9 / 45.6 | 23.3 / 41.8 / 60.5 / 70.0 / 75.0 |

  Inter-token p99 at 32 users of 128/128: 1126 ms at `int8`, the reference 178.
- **Open against the gate** (TENSOR-SPLIT, section 8: the reference's ROCm tensor split at every width, llmx at `--dtype int8`), each with its recovery:
  - width 2, one user, every load (27.4 against 29.9, 19.0 against 21.4, 20.2 against 23.3) and `tg128` (29.3 against 33.6): decode's sums, 54 us each against a budget of about 20; recovery: a wait inside a submission, for which the driver's mapping type is not enough (above) and the kernel route waits on the user, or faster decode a card; the 54 us are the cards waiting on each other's fence, the host being ahead of them at width 2 (about 95 us of submissions and sync files a sum against 212 us of device work a part), so no host change shortens it;
  - width 2, 64 users of 128/128 (150.9 against 192.1), and 32 and 64 users of 1024/128 and of the mixed set (45.8 and 44.6 against 48.0 and 50.0; 67.2 and 64.5 against 70.0 and 75.0): a pass's sums again, which do not shrink as a pass widens, and a prompt's copies; recovery: the same, and step 7's overlap for the copies; `pp512` 526 against 570 goes with the copies (624 with none);
  - width 2, inter-token p99 from 16 users (1126 ms against 178 at 32): a prompt chunk of 512 rows in a pass holds its decode rows for the chunk; recovery: assembly by predicted stage time (phase 3, step 9), which sizes a chunk to the latency the decoders are promised;
  - ahead at width 2: 4 and 16 users of 128/128 (82.3 and 151.0 against 76.8 and 106.7), 16 users of 1024/128 and of the mixed set (45.0 against 42.6, 69.7 against 60.5), level at 32 users of 128/128 (165.8 against 165.6);
  - width 4 and two stages of two: every cell, measured so far at the default precision only; recovery: a group's stage recorded off the scheduler's thread (`feat/tp-stage-threads`, designed and approved) for stages of groups, the host's share of a width-4 sum (four submissions and twelve sync files, about 290 us against 124 us of device work a part, so the host paces it: one sync file a member merged for its peers, and the members' submissions made side by side), two shots for a width-4 prompt's copies (`perf/tp-twoshot`, measured), then what width 2 needs;
  - width 3: not formed on any Qwen3 or Qwen3.5 file; recovery: uneven shares (proposed, after the stage threads);
  - width 8 and eight cards: not measured; the cap of 4 stands until they are.
  The decode budget of a sum, Qwen3-32B Q8_0 under one root, `bench --p 512 --n 128`, two runs (a8b0c910 and d8824ff0; "no sum" is a timing-only build):

  | width 2 | pp512 | tg128 | no sum: pp512 | no sum: tg128 | reference |
  |---|---|---|---|---|---|
  | default precision | 357.5 | 29.6 | 402 | 36.6 | 570 / 33.6 |
  | `--dtype int8` | 526.5 | 29.3 | 624 | 36.8 | 570 / 33.6 |

  At `int8`, the gate's arm, a prompt reads at 92 percent of the reference and would pass it with its copies overlapped (624 with none); decode does not move with `int8` (29.3 against 33.6): a token takes 34.1 ms, 27.2 with no sum, the reference 29.8, so its 128 sums may take 2.6 ms, about 20 us each, against the 54 they take through sync files. Decode at width 2 therefore needs the wait inside a submission, or less device time a token; `int8` does not close it.
  At width 4 the reference is not measured on these cards yet (its width 3 gave 43.7 tok/s); the group's token with no sum takes 15.9 ms, so against a reference of 45 to 50 tok/s a sum may take 30 to 50 us, against about 350 now.
  The wait inside a submission, tried on a RADV patched in a container to map an imported host pointer uncached (Mesa 25.0.7, one call in `radv_amdgpu_winsys_bo_from_ptr`, nothing on the host changed, the source tarball's hash not independently verified): the kernel takes the uncached mapping, and the flag wait of `llmx-vk-handoff exchange` over host inboxes still never sees its peers' flags, at widths 2 and 4, 20 KB to 10 MB, both scopes, three runs, with the probe's spin raised to 2^20 reads; those reads take about 5 ns each, a card's cached read, so the mapping type alone does not make a read of host pages uncached on gfx906. A first run with the shorter spin showed one case arriving, with timeouts and wrong sums; it did not repeat and is not a result. The route is closed as tried.
  Width 3 is refused on every Qwen3 and Qwen3.5 file, whose 8 KV heads three members cannot share whole, where the reference runs it.
- **Gotchas:** the plan's sum term in the pass cost moved to phase 3's step 9, assembly by predicted stage time, which is not built and is where it is consumed; the cards beyond GPU[2] and GPU[3] are lent by their owners for windows posted in the devlog, GPU[0] and GPU[4] while XDEV is away (the user, 2026-10-05).

## The tensor split's kernel route, read in source (2026-10-05, branch docs/tensor-split-kernel-route, docs only, lands by fast-forward)

- **Done:** `docs/TENSOR-SPLIT.md`, section 8, records the reading by file and line (RADV already exports with explicit sync; the kernel's import copies four flags and never that one, shares the exporter's reservation and so always waits on the other card's fences; an uncached mapping is taken for imports; the reference's flag buffer is uncached on both cards) and two candidate patch sets with what each would prove and its chance upstream: forcing explicit sync on imports of KFD memory, and the all-Vulkan route of one kernel line and one RADV line, which is the one to test first; the probe's success criterion; and what reading could not settle.
- **Checks:** a docs change: the suite's docs and dead-code components.
- **Left:** nothing: the user closed the route on 2026-10-07 (`docs/TENSOR-SPLIT.md`, The kernel route is closed), so there is no test to place.

## A fitted budget waits for the room beside it (2026-10-05, branch fix/fit-settle-whole, lands by fast-forward)

- **Cause:** `fitted_kv` asked `settle` only whether a reading held the options' KV budget alone. On one device that reading is taken at once, so a start whose budget fitted on the first, low reading never read again, then cut its checkpoints from it or refused the drafter room beside it: Qwen3.6-27B-MTP Q8_0 at `--ctx-size 8192 --max-seqs 8` with its MTP blocks was refused at 6208 and at 7616 tokens where a settled card gives 8192 and 4 checkpoints. Over several devices the fit already read until the memory had stayed level (`level_first`).
- **Done:** the fit takes its first reading only where it holds the whole request, the options' budget beside every checkpoint and mark asked for and the embedded drafter, and otherwise the reading `settle` ends at: one that rises to hold the whole request, or the last after five quiet seconds. `23bd7671`, the test, fails without it: in `arch-qwen35` a device whose first two reads hold the 512-token budget alone takes 4 checkpoints beside 384 tokens, and one whose first two hold the budget and one mark takes 1 mark of the 6 asked for.
- **Gotchas:** a start whose devices cannot hold every checkpoint and mark asked for beside the whole budget now waits the five quiet seconds, as a start whose budget falls short already did; at default flags the budget is the model's context, which one card does not hold, so such starts waited already. `arch-qwen35`, whose fits wait so on many of its devices, now takes about 74 s on the MI50 machine's CPU, nearly all of it sleeping, and CTest gives it 120 s; the hosted Windows job, which ran 16.8 to 19.1 minutes on main, went past its 20-minute limit with it and now has 25 (`docs/CI.md`).

## The age-limit check follows the files it waits for (2026-10-06, branch fix/disk-age-wait, lands by fast-forward)

- **Found:** `server-resume`'s age-limit check failed in hosted UBSan jobs with "1 entry files left", on branches that cannot reach it.
- **Cause:** the test, not a window in the tier. An entry leaves the index and its file is deleted in one step under the scheduler's lock (`forget_disk`, `DiskStore::evict`), so no reader sees a file the index has dropped. But a copy whose entry the limit deleted is unwritten again and the writer writes it anew, so a new file may land at any moment; the test waited for the index to read zero and then counted every file in the directory, an instant later.
- **Done:** the check takes the files the six turns left and waits, bounded as the other disk checks are, until both are gone.

## The kept-entries check's first turns pass 449 tokens (2026-10-06, branch fix/keep-check-split, lands by fast-forward)

- **Found:** the suite's `server` component failed on one MI50 at main: in the kept-entries check (`check_disk_keep`), the third conversation's follow-up reused 0 tokens after the restart.
- **Bisected** on one MI50 with `--device vulkan:0 --only server`: `235375a9` and `a57b4e50` pass, `16bd1d6a` (disk tier step 5, which added the check) and every head since fail, so the check has failed on an MI50 since it landed; the server did not change under it.
- **Cause:** the expectation, not the server. On a device a longer prompt forks a prompt's rows only where both take one tile split, from 449 tokens on; the check's third first turn is 442 tokens, so its follow-up of 552 forks nothing, as a prompt of another split does everywhere (`server-resume`, fork within a class). On the CPU every prompt is one class, so the check passed there.
- **Done:** `954c7e03`, the test, requires each first turn to pass 449 tokens and fails on the CPU at main, so a hosted job sees it; the fix takes 2300 characters a conversation where it took 2000.
- **Why the gates missed it:** step 5's device suite ran on the Radeon VII alone, where the check does not run (it sends signals, and returns at once on Windows), and its MI50 machine gate ran the `server` component on the CPU. The merge gates now ask a server or scheduler change for the `server` component on an MI50 (`AGENTS.md`, Merge gates).
- **Production** serves a hybrid model over a two-card layer split and is not affected: nothing in the server is wrong or changed, and a model that keeps a state reuses a prefix through its checkpoints, at whole blocks, by the same row classes as before.

## The 8-bit twin block-major where the tile reads it (2026-10-06, branch perf/x8-block-major, lands by fast-forward)

- **Goal:** the second bit-identical lever on the open speed gate (`--dtype int8` block below): the 8-bit tile read a block's 64 columns a row width apart.
- **Done:** a producer told that the tile reads its output next, in a pass of 32 rows or more, writes the 8-bit twin by block and then by column; the tile reads either order from one build (`docs/VULKAN.md`). Row kernels, the Q8_0 decode kernel, passes under 32 rows, the 16-bit twin and every value are unchanged, and no build, variant or flag is added.
- **Result** on one MI50 at default clocks, main 3d8f57f8 and the branch at 45cfa949, before its rebase onto the server change that landed meanwhile, arms in turn, three rounds, medians, tok/s, the fork in the same session on the same card:

  | model, dtype | cell | main | branch | change | fork |
  |---|---|---|---|---|---|
  | Qwen3-8B Q8_0, int8 | pp512 | 1295.8 | 1308.4 | +1.0% | 1313 |
  | Qwen3-8B Q8_0, int8 | pp2048 | 1159.4 | 1169.1 | +0.8% | 1257 |
  | Qwen3-8B Q8_0, int8 | pp4096 | 1012.1 | 1019.8 | +0.8% | 1167 |
  | Qwen3-8B Q4_K_M, int8 | pp512 | 1144.3 | 1184.0 | +3.5% | 820 |
  | Qwen3-8B Q4_K_M, int8 | pp2048 | 1036.3 | 1068.9 | +3.1% | 798 |
  | Qwen3-8B Q4_K_M, int8 | pp4096 | 916.6 | 942.3 | +2.8% | 760 |

  Under f16, which the change does not reach, the same binaries read level (Qwen3-8B Q8_0 842.6 and 842.4, 782.7 and 782.8, 711.9 and 711.2; Q4_K_M within 0.2 percent), and decode is level under both (78.1 and 100.9 tok/s at tg128). Bytes against main: the last 64 logits rows of a 3255-token text and perplexity over 4096-token windows on Qwen3-8B Q8_0, Qwen3-8B Q4_K_M and Qwen3.6-35B-A3B Q4_K_M under int8 and f16, 12 of 12 the same.
- **Left:** Qwen3-8B Q8_0 int8 prompts are level with the fork at pp512 (1308 against 1313) and still behind at pp2048 and pp4096 (7 and 13 percent), which is attention (block above).

## The attention tile addresses its staged words directly (2026-10-06, branch perf/attn-tile-address, lands by fast-forward)

- **Measured first** on one MI50, Qwen3-8B Q8_0, `--dtype int8`, the device time of `attention_tile_kv16_x8` from `bench --profile`, probe branch `exp/attn-tile` (not for merge, forms switched at pipeline creation in one binary): 301 ms at 2048 tokens and 1173 ms at 4096. The driver builds it with 84 registers, 18432 bytes of shared memory and 3 waves a SIMD, and with phase two removed it reads 168 and 652 ms, with phase one's dots removed 156 and 599, with both 17 and 58: the two phases share the time about evenly and the kernel is bound by its instruction count. In phase one a lane ran 748 instructions a step for 256 products, 190 of them an add, a shift down and a shift up before every read of the staged tile.
- **Done:** the staged tile is indexed by word, `token * (DIM / 4) + 8 * run + lane`, and the lane taken as `lid & 7`, so the driver knows the lane's range and folds each address into its read (`shaders/attention_tile.comp`). No operand, order or operation changes.
- **Identity**, candidate 8aac8df2 against main 72309913, the bytes of both arms: on one MI50, the last 64 logits rows of a 3255-token text, perplexity over 4096-token windows of a 14898-token text and 64 greedy ids after the long prompt, on Qwen3-8B Q8_0 (128-wide heads) and Qwen3.6-27B Q8_0 (256-wide heads, the `_d256` builds), each under f16 and int8, 12 of 12 the same (the greedy runs compared without their two timing lines); the device tier's cells on Qwen3-0.6B Q8_0 and Qwen3.5-0.8B Q8_0, 14 of 14 on the CPU and 14 of 14 on the MI50. On the Radeon VII under the AMD proprietary driver, Qwen3-8B Q8_0 under f16: the same three outputs, the same bytes.
- **Gates:** CTest 45 of 45 on Linux and 46 of 46 on Windows; the suite on the CPU and on one MI50 with `--require-tools`, every component passing but `raw-blocks`, which fails in that container for lack of numpy as on main, and `baseline`, which skips there; `backend-vulkan --isa` passes on the MI50 (172 representations, 122456 decode columns under f16 and under int8, 7469 pairs of extents) and on the Radeon VII from a fresh directory (93 representations, 122456 decode columns, 12125 pairs).
- **Timing**, one MI50 (GPU[1]) at default clocks, the card below 55 C before each run, both arms built the same way from detached clones at their own commits (`llmx 0.1.0+g7230991367e2` and `0.1.0+g8aac8df2535d`), arms in turn, three rounds, medians, tok/s; host load 12 to 41 during the runs. The change is to a device kernel and shows in its own device time, so the host's code layout is not in question. The fork column is the one-session table's below, another session on the same card.

  | model | cell | main | candidate | change | fork |
  |---|---|---|---|---|---|
  | Qwen3-8B Q8_0, int8 | pp512 | 1288.0 | 1295.9 | +0.6% | 1310 |
  | Qwen3-8B Q8_0, int8 | pp2048 | 1139.2 | 1159.6 | +1.8% | 1256 |
  | Qwen3-8B Q8_0, int8 | pp4096 | 983.1 | 1012.8 | +3.0% | 1167 |
  | Qwen3-8B Q8_0, int8 | tg128 | 76.7 | 77.1 | level, the spread of both arms is 74.9 to 78.3 | 67.9 |
  | Qwen3-8B Q8_0, f16 | pp512 | 840.5 | 844.4 | +0.5% | |
  | Qwen3-8B Q8_0, f16 | pp2048 | 774.3 | 783.5 | +1.2% | |
  | Qwen3-8B Q8_0, f16 | pp4096 | 697.2 | 712.2 | +2.1% | |
  | Qwen3.6-27B Q8_0, int8 | pp2048 | 392.4 | 398.8 | +1.6% | 368 |

  The tile's device time: 301 to 269 ms at 2048 tokens and 1173 to 1050 ms at 4096, 10.5 percent. Radeon VII, Qwen3-8B Q8_0 f16, three rounds in turn, main and candidate: pp512 341.4 and 345.3, pp2048 318.4 and 324.4, pp4096 292.1 and 300.5 tok/s, every candidate run above every main run.
- **What it does not do:** it closes neither open cell. Other forms that keep F32 were measured and gain nothing: reads grouped per token 270 and 1053 ms, phase two's loop cut in four 279 and 1086, the tile staged as halves and converted at use 299 and 1162. The reference fork's attention (`fattn-tile.cuh` at f58b9f250, read only) multiplies half pairs two an instruction, with an F32 sum in K.Q through an instruction Vulkan does not expose and a half-float sum over V; the cost probes of 16-bit forms and the question of what attention computes in are in the devlog of 2026-10-05 and wait on the user.
- **Review and landing:** reviewed by D2CDEV without findings. The gates above ran on main 72309913; the branch was then rebased onto main 020f9fc9 behind the row-class test and the `--isa` rule, without a conflict in code, and lands by fast-forward once the builds, CTest, the linked dead-code check and the hosted run pass at the rebased head.
- **Gotchas:** the word index and the masked lane were changed and measured together; which of the two the driver needs to fold an address is not separated, so a later edit of either is checked against the kernel's device time under `--profile`; the driver's static instruction count does not show it, since it unrolls the cheaper loop further.

## `--dtype int8` at every row count (2026-10-05, branch feat/int8-prompt, option C step 2)

- **Done:** the budget, frozen before any int8 path ran and unchanged by the new design, since its reference already rounds every row: `tests/int8.py`'s 512-token Q8_0 fixtures (dense untied, dense tied, qwen3moe; texts of 160, 200 and 256 tokens, windows of 128 and 256) at their own weights and four more seeds, HF float32 on each file's decoded weights, every matrix product's input but the routers' rounded by the twin's rule at 127 levels, the output head included (`tools/calibrate_dtype.py`, class `int8`): at most 0.0395 logits (dense tied, seed 3) and 0.00205 NLL (dense tied, seed 0), budgets 0.0790 and 0.00409 in `tests/data/dtype_budget.json`; the F16 and BF16 classes reproduced every frozen field in the same run (torch 2.5.1+cpu, transformers 4.55.2, one thread, the HF reference container, cores 4 to 7). For comparison F16's budget is 0.0033 logits.
- **Built:** `Dtype::int8`; the 8-bit builds (`LLMX_I8`) of every kernel a quantized type but MXFP4 takes under f16 on an integer-dot device, chosen through `WeightKernels::int8_row`, `int8_tile` and `int8_tall_tile`: the row kernels of the Q4 and K-quant families (`matmul_row.comp`, the builds dtype step 4 removed, ported onto today's kernels, Q6_K's -32 taken into each weight byte), the Q8_0 decode kernel (`matmul_vec_q8.comp`, every `kVecBuilds` build, its lanes, blocks and order the 16-bit kernel's) and the prompt tile (`matmul_tile_q.comp`); the 8-bit twin after the 16-bit one in the same scratch (`xquant8_base`, `x8_base_bytes`), written by every producer's second build (specialization constant 7, `TWIN8`) once a matmul has read it (`want_x8_`, `XqTag::has8`), lane per value (`xquant8_block`) or word per lane (`xquant8_word`); `VulkanBackend::native_dtypes` lists int8 where the integer dot is preferred; `resolve_dtype` widens a device without it to F16 where it has F16, else F32, naming that dtype in the record (`fallback to f16`), and a run whose every device widened to one dtype is that dtype's; `--dtype int8` in the CLI, `llmx-split-check`, `run_tests.py` and `tools/long_context_check.py`; the witness `block-int8`. The threshold (`kInt8PromptTokens`, `row_dtype`), the refusal (`check_dtype_request`, `placement_host`), the separate 8-bit quantizer (`quantize_x8`) and the suite's skips under int8 (`common.int8_cpu_skip`) are gone.
- **Checks:** `dtype` holds auto never choosing int8, a placement of int8 devices resolving to it, and the widening per device, F16 on the CPU and on a device of F16 and F32, F32 on a device of F32 alone, named in the record with the warning; `backend-vulkan` holds Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K rows 512 and 1280 wide under int8 within 1e-3 of the CPU on the same 8-bit activations at 1, 3, 8, 13 and 64 generated tokens and prompts of 40, 64, 128 and 200 tokens (885920 outputs, chosen away from rounding ties: with ties, the device's reciprocal of a block's peak, whose last bit can differ from the host's, rounded one value a step apart and moved single outputs by up to 0.03), witnessing `block-int8`, a prompt's bits as one call and as two slices, F32 and MXFP4 under int8 giving f16's bits, every decode column of every 8-bit build the same column alone (122456 columns, as for f16), and the weight dispatch naming each 8-bit kernel; `vulkan-quantization` the 8-bit twin's reconstruction and sums from both quantizers' second builds, their 87820 words identical, and the 16-bit twin beside it the first builds' words; the `int8` component the fixtures under f16 and int8, batched and one token a step.
- **Smoke** at e1d101eb on one MI50 (GPU[1]), `bench --model` pp512/tg128, two runs, f16 then int8: Qwen3-8B Q8_0 846.0/77.5 and 1291.4/78.2, Qwen3-8B Q4_K_M 787.0/88.6 and 1139.6/99.4 tok/s; greedy replies to "The capital of France is" begin "Paris." under both; CTest `dtype`, `vulkan-quantization`, `vulkan-buffer`, `vulkan-lifetime` and `backend-vulkan` pass. The default builds of `matmul_row.comp` and `matmul_vec_q8.comp` are main's SPIR-V byte for byte; the producers' modules differ by the specialization constant.
- **Gates** at 6c7d445f against main 8f08e409 (`llmx 0.1.0+g6c7d445f113d`, 13b1ec61a75b39cb; `llmx 0.1.0+g8f08e4095314`, 42b2b33c52d5b6fd), GPU[1], cores 4 to 7: CTest 45 of 45; the suite on the CPU and the MI50, each under the default dtype and under `--dtype int8`, passes every component but raw-blocks, which needs numpy the image lacks; the Qwen3-0.6B and Qwen3.5-0.8B Q8_0 identity cells 14 of 14 on the CPU and 14 of 14 on the MI50, so f16 keeps main's bits; `llmx-backend-vulkan-test --isa` passes on the MI50 with 172 representations, the 8-bit builds' counts among them; the linked dead-code check 14 findings, all listed. The tiny HF gate under int8 on the MI50, batched and one token a step: logit/NLL errors 0.0312/0.00185 (dense untied), 0.0276/0.00058 (dense tied), 0.0179/0.00106 (MoE) against the budget's 0.0790/0.00409, headroom 61, 65 and 77 percent of the logit budget; f16 0.0001/0.00001.
  The Radeon VII under Windows and the AMD proprietary driver at 912894ca: CTest 46 of 46; `--dtype int8` there reads `dtype: int8 -> f16 (model declares bf16); vulkan:0 fallback to f16: ...` with the warning and gives f16's 48 greedy ids on Qwen3-8B Q8_0; `--isa` passes. At 6c7d445f `--isa` failed there in the Q4_K row kernel's replay check, where main passed, and passed at 912894ca with the specialization entries in constant order; the order was taken for the cause and was not (next line).
  **Correction (2026-10-06, branch fix/isa-fresh-path, lands by fast-forward):** that failure is the driver's, on any tree. On the Radeon VII (AMD proprietary driver 26.5.2) `llmx-backend-vulkan-test --isa DIR` fails with "a generated row of a 40-row replay beside a prompt differs from it decoded alone" on a Q4_K matmul whenever a run without `--isa` of the same executable path came before it, and passes from a path no such run used. Reproduction at main 72309913: the executable copied to a new directory and run with `--isa` passes, three copies twice each; copied to another, run plain, then with `--isa`, then plain, then with `--isa`, it passes, fails, passes, fails. Main rebuilt with the old entry order (constant 7 listed after 13) behaves the same: `--isa` passes from two new directories, twice each, and fails after a plain run. So the entry order changes nothing, and what the earlier run met was a build directory CTest had run in. The order stays the constants', which is harmless, the comment no longer names the driver, and the hand rule for `--isa` on that card now asks for a fresh directory (AGENTS.md, Tests). Not checked: whether `--profile`, which opens the backend for diagnostics too, is affected on that card. Gates: the comment is the only change to a source, so CTest and the hosted run; `docs` and `dead-code` pass.
- **Real models against HF** at 912894ca on GPU[1], `tests/baseline.py` and `tests/baseline_qwen35.py` at 512-token windows: under f16 every cell passes. Under int8 every NLL cell passes and two rankings fail, recorded rather than special-cased (the user's decision): Qwen3-0.6B Q4_0's "The capital of France is", top-5 overlap 3 of 5 against its bound of 4, HF's top five 12095, 7407, 279, 1112 and 30743, where f16 keeps 7407 at 13.575 just above 320 at 13.532 and int8 puts 320 at 13.663 above 7407 at 13.501, the cell an 8-bit tied Q6_K head failed in September; and Qwen3.5-0.8B Q8_0's `chat-00`, top-5 overlap 4 of 5 against 5, HF's fifth and sixth 332 at 17.959 and 9175 at 17.656, 0.30 apart, which int8 orders 9175 at 17.772 above 332 at 17.481, the cell 8-bit prompt rows missed in September. Qwen3-0.6B Q8_0, Q5_K_M and Q4_K_M and Qwen3.5-0.8B Q4_K_M (114 checks) pass under int8.
- **Routed models**, the last 512 positions of a window of wiki.test.raw (563 and 575 tokens, measured on the every-row design at f79cf9e44, whose shaders are the landed ones; a single prompt of that length took the 8-bit tile at every row in the first design too, which is why that design's figure on Qwen3.6-35B-A3B Q4_K_M was the same): Qwen3.6-35B-A3B Q4_K_M on one MI50, file-exact continuous NLL 2.00184 under int8 and 2.01038 under f16 against HF's 2.01045, top-1 int8 against f16 504 of 512 (f16 against f32 510), top-8 expert sets differing from f32's in 17.35 percent of 22520 routings under int8 and 1.48 percent under f16; Qwen3-30B-A3B Q8_0 split over GPU[1] and GPU[4], which has no HF golden since its F32 reference does not fit the host: NLL of the same text 2.36369, 2.37508 and 2.37495 under int8, f16 and f32, top-1 int8 against f16 506 of 512 (f16 against f32 511), routings differing from f32's in 9.50 percent of 27600 under int8 and 1.14 percent under f16.
- **Timing** at 912894ca against main 8f08e409 (`llmx 0.1.0+g912894ca2816`, 9aa34694b75a7e06; `llmx 0.1.0+g8f08e4095314`, 42b2b33c52d5b6fd), one MI50 (GPU[1]) at default clocks, the card below 55 C before each run, `bench --model` two repetitions, arms main f16, int8, branch f16, int8, main f16 (tg128 after the 512-token prompt); the reference's server on the same card, one request at a time, cache off, three each, its tg128 after a 64-token prompt; tok/s:

  | model | cell | main f16 | branch f16 | int8 | reference |
  |---|---|---|---|---|---|
  | Qwen3-8B Q8_0 | pp512 | 845.5, 840.7 | 841.3 | 1288.0, 1287.2 | 1309, 1303, 1321 |
  | Qwen3-8B Q8_0 | pp2048 | 778.1, 774.6 | 774.4 | 1126.2, 1139.5 | 1256, 1256, 1255 |
  | Qwen3-8B Q8_0 | tg128 | 76.93, 75.54 | 75.02 | 77.94, 76.15 | 68.7, 69.0, 69.1 |
  | Qwen3-8B Q4_K_M | pp512 | 787.6, 785.8 | 786.0 | 1137.3, 1138.8 | 820, 817, 825 |
  | Qwen3-8B Q4_K_M | pp2048 | 728.0, 727.3 | 727.4 | 1006.5, 1019.9 | 799, 799, 799 |
  | Qwen3-8B Q4_K_M | tg128 | 87.48, 87.25 | 84.98 | 101.19, 98.85 | 83.8, 84.3, 84.3 |
  | Qwen3.6-27B Q8_0 | pp512 | 261.0, 260.6 | 260.5 | 407.5, 407.3 | 357, 346, 355 |
  | Qwen3.6-27B Q8_0 | pp2048 | 253.5, 254.1 | 253.9 | 392.7, 392.4 | 353, 361, 338 |
  | Qwen3.6-27B Q8_0 | tg128 | 23.63, 23.54 | 23.43 | 23.85, 23.63 | 20.3, 22.1, 20.9 |
  | Qwen3.6-35B-A3B Q4_K_M | pp512 | 1133.3, 1133.3 | 1132.3 | 1583.3, 1582.4 | 514, 914, 1007 |
  | Qwen3.6-35B-A3B Q4_K_M | pp2048 | 1098.6, 1098.6 | 1098.5 | 1506.5, 1506.8 | 1061, 1023, 1065 |
  | Qwen3.6-35B-A3B Q4_K_M | tg128 | 110.41, 109.35 | 105.18 | 117.04, 110.52 | 83.1, 83.8, 83.9 |
  | Qwen3.5-9B Q4_K_M | pp512 | 780.3, 779.5 | 779.6 | 1122.0, 1122.2 | 723, 630, 724 |
  | Qwen3.5-9B Q4_K_M | pp2048 | 761.0, 760.9 | 761.1 | 1085.0, 1084.2 | 752, 747, 750 |
  | Qwen3.5-9B Q4_K_M | tg128 | 81.41, 80.66 | 79.15 | 90.90, 91.94 | 71.0, 71.3, 71.3 |

  The branch's one f16 decode arm ran 1 to 4 percent below main's two, so f16 decode was timed again on its own: six interleaved rounds of `bench --model --p 64 --n 128 --r 3`, medians branch against main 88.37 against 88.17 (Qwen3-8B Q4_K_M), 110.18 against 110.17 (Qwen3.6-35B-A3B Q4_K_M) and 81.40 against 81.25 (Qwen3.5-9B Q4_K_M), level, every sample within 1 percent; the branch's f16 prompts are main's. int8 against f16: prompts +45 to +57 percent, decode +14 percent on the dense Q4_K_M files (8B, 9B), +1 to +4 percent on the Q8_0 files and the routed file. Against the reference, int8 is ahead in every cell but the 8B Q8_0's prompts, level at 512 (1288 against 1303 to 1321) and 10 percent behind at 2048.
  No run was dropped; the only activity record is the host's load average beside each run, 6 to 43 from other work on the machine (`i8btime-912894ca2`, `i8bab-912894ca2` on the MI50 machine), no per-process or GPU monitor ran.
- **16k depth**, Qwen3.5-9B Q4_K_M on GPU[1], the three frozen histories of the f16 depth check (16384 prompt tokens, 511 forced reply tokens as decode steps, the last 512 rows scored; `exp/int8b-depth` d0e2c461 prints each row's top 50). The int8 reference was predeclared from the arithmetic int8 reaches: HF on the file's own weights with the input of every Linear, the head included, rounded by the frozen int8 rule at every row and F32 sums (`tools/calibrate_dtype.py`, class `int8`; every Linear of this file is quantized, and it has no router), 80 and 45 minutes on four cores. Rows are exact, an accepted tie (HF's second token with HF's top two within 0.1) or a miss:

  | history | llmx | reference | exact | accepted tie | miss | largest error on HF's top five |
  |---|---|---|---|---|---|---|
  | raw, row 8 | int8 | int8 | 505 | 3 | 4 | 8.16 |
  | raw, row 16 | int8 | int8 | 505 | 3 | 4 | 8.16 |
  | chat | int8 | int8 | 510 | 2 | 0 | 0.59 |
  | raw, row 8 | int8 | f16 | 507 | 0 | 5 | 5.76 |
  | raw, row 16 | int8 | f16 | 507 | 0 | 5 | 7.56 |
  | chat | int8 | f16 | 510 | 1 | 1 | 1.72 |
  | raw, row 8 | f16 | f16 | 512 | 0 | 0 | 0.24 |
  | raw, row 16 | f16 | f16 | 512 | 0 | 0 | 0.32 |
  | chat | f16 | f16 | 512 | 0 | 0 | 0.09 |

  So at this depth int8 misses 4 of 512 top-1 tokens on each raw history even against a reference rounding by its own rule, with single logits up to 8 off, where f16 misses none and stays within 0.33: two computations that round to 8 bits part from each other there, since a rounding near a tie in one layer moves the next layer's inputs. The misses' HF top-two gaps are 0.13 to 3.85. This is int8's measured cost at depth, recorded like the two ranking cells; f16's depth check is unchanged.
- **Landed** at c8b2ac8d by fast-forward on 2026-10-05, after the coordinator's review of the final diff, without findings; its hosted run, 37341190895, passed every job. At that head on the MI50 machine: CTest 45 of 45, the suite on the CPU and on one MI50 under the default dtype and under `--dtype int8` passing every component but raw-blocks (no numpy in the image), and the linked dead-code check's 14 findings all listed.
- **Open speed gate** (the user, 2026-10-05: at the same precision llmx must be ahead of the reference in every cell; the reference computes with 8-bit activations, so the matched arm is `--dtype int8`): in the timing table above int8 is behind on Qwen3-8B Q8_0 prompts, 1288 against 1311 tok/s at pp512 and 1133 against 1256 at pp2048, and ahead in every other cell measured. The reference's decode there followed a 64-token prompt and llmx's a 512-token one, so the decode cells are to be measured again matched, on all five models. Recovery work, in the order the coordinator approved: prompt attention at long prompts, then block-major 8-bit activations written by the producers, then the tile's shape; the repacked weight layout was probed and rejected (below).
- **Matched table, one MI50, 2026-10-05, one session** (GPU[1] for every arm, default clocks, the card below 55 C before each run, the arms in turn per model, two rounds; pp512, then tg128 after that same 512-token prompt, and pp2048; tok/s). Each column is labelled with the precision of its matmul inputs, so like is compared with like: llmx int8 with the three 8-bit arms, llmx f16 beside them as another precision.
  - llmx at c8b2ac8d (`llmx 0.1.0+gc8b2ac8d29c1`, d18bb7ff031865b8): `bench --model`, means of the two rounds.
  - mx-llama.cpp (the fork, f2a54df5, ROCm 7.2.1 container, its repacked 8-bit path) and upstream llama.cpp master c25030496079 of 2026-10-05, unmodified, built twice from one clone: ROCm/HIP (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, Release, the same container) and Vulkan (`-DGGML_VULKAN=ON -DGGML_NATIVE=OFF`, Release, Mesa RADV in the llmx Vulkan image). Each as `llama-server -ngl 99 -fa on -c 8192 -np 1 -b 2048 -ub 512`, pinned to the card, one request at a time, cache off, medians of the six requests of the two rounds, a server's slower first request included.
  - Which path the upstream arms took: its Vulkan build logs `int dot: 1` for the card, and with `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` the same build read 368 tok/s at pp512 on Qwen3-8B Q8_0 against 850 without it, so its default prompt path there is the integer-dot one over 8-bit activations (its decode moved from 56.0 to 51.9); its logs name operations, not pipelines, so no kernel name is recorded. Its HIP build's 8-bit MMQ kernels are taken by its dispatch rule, not witnessed.
  - The host's load average was 7 to 21 through the session, from other work on the machine, recorded beside each run in the log; no run was dropped.

  | model | cell | llmx f16 (16-bit) | llmx int8 (8-bit) | mx-llama.cpp ROCm (8-bit) | upstream ROCm (8-bit) | upstream Vulkan (8-bit) |
  |---|---|---|---|---|---|---|
  | Qwen3-8B Q8_0 | pp512 | 844 | 1290 | 1310 | 819 | 821 |
  | Qwen3-8B Q8_0 | pp2048 | 778 | 1142 | 1256 | 804 | 774 |
  | Qwen3-8B Q8_0 | tg128 | 77.4 | 77.9 | 67.9 | 69.0 | 55.1 |
  | Qwen3-8B Q4_K_M | pp512 | 788 | 1141 | 818 | 861 | 718 |
  | Qwen3-8B Q4_K_M | pp2048 | 729 | 1023 | 798 | 835 | 687 |
  | Qwen3-8B Q4_K_M | tg128 | 85.1 | 99.4 | 82.8 | 82.2 | 82.6 |
  | Qwen3.6-27B Q8_0 | pp512 | 262 | 409 | 353 | 226 | 231 |
  | Qwen3.6-27B Q8_0 | pp2048 | 256 | 394 | 368 | 231 | 237 |
  | Qwen3.6-27B Q8_0 | tg128 | 23.6 | 23.8 | 22.0 | 20.9 | 17.3 |
  | Qwen3.6-35B-A3B Q4_K_M | pp512 | 1136 | 1585 | 970 | 932 | 1009 |
  | Qwen3.6-35B-A3B Q4_K_M | pp2048 | 1101 | 1505 | 1064 | 1109 | 1152 |
  | Qwen3.6-35B-A3B Q4_K_M | tg128 | 110.8 | 117.0 | 82.8 | 78.6 | 78.5 |
  | Qwen3.5-9B Q4_K_M | pp512 | 780 | 1122 | 713 | 721 | 646 |
  | Qwen3.5-9B Q4_K_M | pp2048 | 761 | 1082 | 750 | 746 | 677 |
  | Qwen3.5-9B Q4_K_M | tg128 | 81.4 | 92.8 | 70.5 | 69.9 | 72.7 |

  **Open cells:** at the same precision llmx int8 is behind the fork on Qwen3-8B Q8_0 prompts, 1290 against 1310 at pp512 (-1.5 percent) and 1142 against 1256 at pp2048 (-9 percent), and ahead of all three 8-bit arms in every other cell. Against unmodified upstream it is ahead everywhere, by 23 to 77 percent on prompts and 13 to 49 percent on decode. The default f16, a wider precision than any of them, is beside them for reference: ahead of upstream Vulkan in every cell but the routed file's pp2048 (1101 against 1152), and behind upstream ROCm on Qwen3-8B Q4_K_M prompts (788 and 729 against 861 and 835).
- **What the open cells are made of.** Qwen3-8B Q8_0 under int8 at prompt lengths 512, 1024, 2048 and 4096 in the same session (`--profile`): 1284, 1234, 1127 and 977 tok/s, the prompt tile 353, 713, 1415 and 2835 ms and prompt attention (`attention_tile`) 22, 79, 300 and 1175 ms, four times the time for twice the tokens. The fork at the same lengths: 1310, 1304, 1256 and 1167 tok/s. Fitting each as a term linear in the tokens and one quadratic, the linear part is 0.737 ms a token in llmx and 0.750 in the fork, so everything but attention is already faster, and the quadratic part is 1175 ms at 4096 in llmx against about 440 in the fork: prompt attention costs 2.7 times the fork's. With the fork's attention llmx would read about 1334, 1258 and 1185 tok/s at 512, 2048 and 4096 against its 1310, 1256 and 1167. So prompt attention is the lever, and the block-major activations (below) the margin on top.
- **Rejected lever: the repacked Q8_0 weight layout** (`exp/repack-probe`, not for merge; 2026-10-05). A probe wrote each Q8_0 matrix once more as a quant plane of rows at a stride of nin bytes, 16 more where that is a multiple of 128, and an f16 scale plane, as the fork's repack has it, and read it through builds of the Q8_0 decode kernel and the integer tile (16-bit and 8-bit), the wide Q8_0 row build and the float tile; one binary, the layout chosen by an environment variable, bit-identical to the file's layout on both cards. On one MI50, four interleaved rounds, file layout against repacked: Qwen3-8B Q8_0 under int8 pp512 1289 against 1243, pp2048 1135 against 1102, tg128 78.4 against 77.3, decode of 2, 4 and 8 sequences level; under f16 pp512 843 against 835, tg128 level, 8 sequences 309 against 287; the tile's device time 358 against 369 ms (8-bit) and 570 against 574 (16-bit); Qwen3-0.6B Q8_0 decode +2 to +7 percent and prompts -1 to -2. So on the MI50 it gains nothing on the 4096-wide model and costs its prompts 1 to 4 percent, and it is not built. The 7.8 percent an ablation had shown for contiguous weight loads (The 16-bit prompt tile on the MI50, below) did not hold for a correct repacked tile: that ablation also dropped work.
  - A separate candidate from the same probe, for a narrower change later with its own probe: on the Radeon VII under the AMD proprietary driver the wide Q8_0 row build, which loses its half-word shuffles on the repacked layout, decoded 13 to 15 percent faster on Qwen3-0.6B and Qwen3.5-0.8B Q8_0 (203 to 229 and 184 to 211 tok/s) and 16 to 29 percent faster at 2 to 8 sequences, while their prompts through the float tile lost 2.5 percent; Qwen3-8B Q8_0 does not fit twice in that card's 16 GB, so 4096-wide rows are unmeasured there.
  - Block-major 8-bit activations for the tile (the same probe branch): the tile's device time 352 to 347 ms at pp512 and 1413 to 1391 at pp2048 on Qwen3-8B Q8_0, bit-identical; with the probe's extra quantize pass Qwen3-8B Q4_K_M +1.4 and +2.2 percent end to end. It follows attention, written by the producers.
  - The K-quants get a layout probe only if attention and the activations leave their cells open.
- **To watch:** in the first device suite at c8b2ac8d the `server` component's MXFP4 subcheck failed once with an empty error message; the component passed twice when run again alone on the same card, and had passed at 6c7d445f on another. The card held 14.9 GB of another process's memory when read after the reruns. This is an observation, not a proven cause: contention on a shared card is one reading, and the subcheck is to be watched in later device suites.
- **Row classes under int8** (2026-10-05, branch test/int8-row-classes, tests only, lands by fast-forward): D2CDEV's review of the landed c8b2ac8d found that the row-class identity check (`tests/row_classes.hpp`) ran for f32, f16 and bf16 only, so no pair of extents of one class was compared under int8. It now takes int8 where the backend lists it; on one MI50 `backend-vulkan` reads 9933 pairs of extents of one class with the same bits against 7469 before, all 2464 new pairs equal, and CTest passes 45 of 45 on Linux. Reviewed by D2CDEV without findings.
- **Gotchas:**
  - (accepted by the coordinator, 2026-10-05) the MoE fixture routes every token to all eight experts: with top-3 routing, the smallest gap between a token's third and fourth router logit over the fixture's 1128 tokens was 1e-4 to 1e-2 at every seed from 0 to 80, so any rounding could turn a routing over and set the budget by a routing rather than by the products' precision; the switch was made on those gaps, before any MoE int8 error had been computed (the dense rows had run, and were unchanged by it).
  - The first design (prompts of 128 tokens or more on the 8-bit tile, everything else f16, refused where it could not run) was built and gated at e3df0c80 and 7973149d before the user's decision; its measurements are history, not evidence for this design: prompt rates on one MI50 at pp512 of 1281 (Qwen3-8B Q8_0), 1131 (Qwen3-8B Q4_K_M), 406 (Qwen3.6-27B Q8_0) and 1584 tok/s (Qwen3.6-35B-A3B Q4_K_M) against f16's 841, 785, 260 and 1133, and on Qwen3.6-35B-A3B Q4_K_M 8-bit prompt inputs turned 17.35 percent of top-8 expert sets over against f32, f16 1.48 percent, with top-1 agreement against f16 at 504 of 512.
  - Its 16k HF reference, prompt rows int8 and reply rows F16, was stopped after 46 minutes on the decision, being void for this design.

## The 16-bit prompt tile on the MI50 (2026-10-05, option C step 1, measured, no code landed)

- **Profile** at main 1cb7a2a4 on one MI50 (GPU[1]), device time of one prompt: the integer-dot tile (`matmul_tile_q.comp`) is 93 and 85 percent of Qwen3-8B Q8_0 at pp512 and pp2048 (848 and 777 tok/s), 76 + 17 and 71 + 16 percent (the Q4_K tile and the Q6_K tile) of Qwen3-8B Q4_K_M (787 and 730), 93 and 91 percent of Qwen3.6-27B Q8_0 (262 and 255); attention is 4 to 11 percent, the 27B's delta rule 3.
- **Where the tile's time goes**, Qwen3-8B Q8_0 pp2048, tile 2251 to 2265 ms, by ablations with wrong results (`exp/tile-abl-a`, `-b`, `-c`): staging only the first step 1619 ms, staging from computed values with no global loads 1819 ms, no per-block scale arithmetic 1962 ms. So the global loads cost about 19 percent, the barriers and shared-memory writes about 9, the scale arithmetic about 13 (three operations per row, column and block, fixed by bit identity), and the dots with their shared-memory reads run at about 81 percent of the two-wide 16-bit dot peak.
- **Rejected candidates**, each bit-identical (logits of 300 rows on Qwen3-8B Q8_0 and Q4_K_M byte for byte) and slower: activation columns padded off the shared-memory banks (`exp/tile-xpad`, +3 to +4 percent tile time on all three models); the step loop unrolled by two with the next step's loads in flight during the dots, in five forms (`exp/tile-pipe2`, `-pipe2b`, `-pipe2c` and their no-break and branch-free forms, +1.5 to +3.8 percent), since the tile already holds 128 VGPRs for two waves a SIMD (90 needed before scheduling) and holding the next step's registers across the dots makes ACO add 100 to 300 moves and sink the loads back to their use; one block a step (`exp/tile-q8-step1`, +7.5 percent, still 128 VGPRs); the 64-row tile at three waves a SIMD everywhere (`exp/tile-short-only`, +14 percent); workgroups grouped by row tiles so the running ones share weights (`exp/tile-order-g8`, `-g4`, `-g1`, +4 to +32 percent).
- **The measured lever for the default path: a repacked Q8_0 layout.** The tile reading its weights as if stored tile by tile, block kb of a tile's 128 rows adjacent (`exp/tile-abl-d`, wrong values, same bytes per load), took the tile from about 2258 to 2081 ms (-7.8 percent): pp512 843 to 910 and pp2048 777 to 831 tok/s on Qwen3-8B Q8_0. The scattered 68-byte chunk each row reads per step is the cost, not the weights' reuse. Realising it needs Q8_0 matrices stored in that layout: either the decode kernels read the same layout, a rewrite of the Q8_0 decode builds, which are held bit for bit and batch-invariant, or a second copy for the tile, twice the memory of every Q8_0 matrix, which the 27B cannot afford; the K-quants would each need their own layout. To be designed later together with the decode kernels; not now (the coordinator, 2026-10-05).

## The synthetic bench times its steps warm (2026-10-05, branch fix/bench-warm-up, lands by fast-forward)

- **Bug:** the suite's synthetic prefill floor (1000 tok/s) missed inside the full suite on the Radeon VII four steps in a row (619 to 869 tok/s) and passed alone (about 2000). The bench timed a fresh model's first 64 steps as its prefill, so one-time setup landed in it: on the Radeon VII 8 timed steps took 8.3 to 8.6 ms against 0.39 ms a step steady, and after more than half an hour with the device idle, as in the full suite, that setup reached 70 to 80 ms; decode, timed after it, never missed. Not a regression: main's build missed the same way in the same run.
- **Fix:** `time_steps` runs the whole prefill and decode once untimed and resets the model before timing them; the floor stays at 1000 tok/s. The first commit moves the timing into `time_steps` unchanged and adds a `cli-output` check that nothing is allocated while the clock runs, over a CPU backend counting its allocations, which fails on it; the second makes it pass.
- **Gates** (tests and tools tier): CTest and the CPU suite on Windows; `perf` on the Radeon VII; the hosted run.

## Disk tier (2026-10-04 to 2026-10-05, DISK-TIER steps 1 to 6, landed by fast-forward)

- **What it is:** a third tier under the device and host tiers ([DISK-TIER](DISK-TIER.md)), off unless `serve --disk-cache-bytes` gives it room: what the host tier would drop next is written to a file of the server's own while it stays in host memory, and a waiting request whose history an entry shares more of than anything in memory has it read back into host memory, bit for bit, then promoted and forked as from the host tier; a clean exit under `--disk-cache-keep` writes what memory holds for the next server of the same build and model, and entries unused past `--disk-cache-max-age` go.
- **Steps on main:** identity and layout (893a46eb), the store (01591f5b), write-ahead demotion (a2f32b8f), restore (a57b4e50), exit, keep and age (16bd1d6a), and step 6: a read takes the store between two 4 MiB chunks of a write, the admission order in `server/policy.hpp` (`admit_waiting`) with disk reads in `server-passes`' random schedules, `/v1/health`'s `disk_ready`, and the docs.
- **Changes to the plan, each approved:** demotion writes ahead of need, since an entry still being written holds its room and write-on-drop would keep nothing (the wear counted as `disk_bytes_written` against `disk_bytes_read`); no separate index file, every entry's header and modification time being one; the hosted UBSan and Vulkan jobs' limits raised to 25 minutes, both already within a minute of 15.
- **Measured**, Qwen3.8-27B Q8_0 on one MI50 at default clocks, four CPU cores, the default host tier, `--max-seqs 8 --ctx-size 32768`, both llmx arms built the same way from detached trees, the reference server at `-np 8 -kvu`; users take turns through 20 turns each, then a regenerate and an edit at turn 2; time to first token p50 / p99:

  | workload | arm | follow-up turns | regenerates | edits | whole re-reads |
  |---|---|---|---|---|---|
  | 24 users, more than the host tier holds | main 9091ee2e | 2.64 / 43.16 s | 5.29 / 8.45 s | 5.28 / 6.64 s | 302 of 504 |
  | | disk tier fcfdf3ec, 64 GiB on `/zpool1` | 2.20 / 4.55 s | 5.25 / 6.26 s | 5.26 / 6.53 s | 115 of 504 |
  | | disk tier fcfdf3ec, 16 GiB on the root NVMe pool | 2.42 / 37.78 s | 5.31 / 6.27 s | 5.29 / 6.57 s | 177 of 504 |
  | | disk tier 38a9d6f5 (reads between a write's chunks), 16 GiB on the root NVMe pool | 2.22 / 8.86 s | 5.29 / 6.28 s | 5.28 / 6.58 s | 139 of 504 |
  | | reference server | 6.69 / 31.40 s | 5.01 / 5.79 s | 1.30 / 1.96 s | 480 of 504 |
  | 6 users, which the host tier holds | main 9091ee2e | 2.35 / 3.71 s | 2.01 / 4.25 s | 3.07 / 4.28 s | 3 of 126 |
  | | disk tier d30e0cb5, 64 GiB on `/zpool1` | 2.35 / 3.69 s | 2.28 / 4.49 s | 3.06 / 4.28 s | 3 of 126 |
  | | reference server | 2.77 / 4.10 s | 5.09 / 6.26 s | 1.61 / 1.96 s | 6 of 126 |

  | arm (24 users) | written ahead | read back | entries read | waits for a read, total |
  |---|---|---|---|---|
  | 64 GiB on `/zpool1` | 257.3 GB | 168.8 GB | 585 | 296, 174.3 s |
  | 16 GiB on NVMe, reads after the whole write | 205.6 GB | 89.2 GB | 277 | 144, 898.4 s |
  | 16 GiB on NVMe, reads between chunks | 276.2 GB | 120.7 GB | 392 | 196, 216.6 s |

  With 24 users the disk tier cuts the follow-ups' p99 from 43.2 s on main to 4.6 s with 64 GiB on `/zpool1`, and leads the reference server's 31.4 s by 7x; edits still trail the reference (5.26 against 1.30 s), the boundary positions of the edited-turn finding, which the disk tier does not change. The root NVMe pool writes at about 330 MB/s, so with 16 GiB the writer is busy almost all the time: reads queued behind whole writes waited 6.2 s each, and taking the store between a write's chunks cut that to 1.1 s and the p99 from 37.8 to 8.9 s. With 6 users, which the host tier holds, the disk tier wrote 11.9 GB, read back 1.1 GB and forked exactly what main forked; the regenerates' 0.25 s there were boundaries released ahead of superseded copies, the order fixed before the 24-user runs.
- **Gotchas:** a server serves without the disk tier until the model file's digest is known and its store's probe done, about twenty seconds for a 27 GB file not hashed before, and `/v1/health`'s `disk_ready` says so; a test of the tier waits for it (the hosted runners reached it only after a test's turns). On Windows a console closed ends the process a few seconds later, so a keep flush started that way may not finish. A request waiting for a read keeps its place while later ones that fit pass it, which `server-passes` now simulates through the scheduler's own `admit_waiting`.
- **Gates of step 6:** CTest and the suite on Windows and the Radeon VII, the disk tier's server checks on the MI50 machine's CPU, the linked dead-code check on the rig, the hosted run. Production turns the disk tier on only on the user's decision.

## Tensor groups on Vulkan (2026-10-04, branch feat/tp-vulkan, step 3 of TENSOR-SPLIT, lands by fast-forward)

- **Done:** `VulkanCollective` and `VulkanBackend::join` (two parities of inboxes exported and imported as dma-buf, a binary semaphore a peer exported as a sync file, `submit_signalling` and `wait_on` in the backend's submission, each peer's wait going to its next submission once every member has submitted, the sum a copy and `add`s in member order), refused by name across PCI root complexes; the head's gather as copies each member enqueues into imported host rows, the pass's logits waiting on every member's ticket, so no member waits on another while it records (the coordinator's review of step 2); a sum that fails part way drains the members and makes its semaphores again, so the passes beside it go on, and `sync` submits waits an exchange left, with `vulkan-lifetime`'s collective cases; `run_tests.py --tensor-width`, and a tensor width in `tests/baseline_8b.py`, `tools/long_context_check.py` and `tools/server_mix_check.py`.
- **Gates** (model, kernel and device tier), the branch at f4054f63 (`llmx 0.1.0+gf4054f639880`, binary sha256 1654a28a63c42ef3) against main 1c65c448 (`llmx 0.1.0+g1c65c4482090`, 6e885c1818f6a0b1), both built with Vulkan the same way from detached worktrees in the development image, cores 8 to 11, two MI50s of one root complex (GPU[2] and GPU[3], PCI 83:00 and 86:00), RADV, default clocks:
  - numerics of a group of two MI50s: `backend-vulkan`'s collective check, six sums in member order bit for bit with its refusals; the tensor-split fixtures within 2.8e-7 of HF and passing the device-reference criterion against one card with 64 greedy steps; the HF gate on the Qwen3-0.6B gate files, the qwen35 files skipping as refused until step 5; the Qwen3-8B Q8_0 HF check, 41 checks, NLL within 0.0016 of HF against 0.01; the 16k long-context check on Qwen3-8B Q8_0, two runs the same and one card's top choice at 512 of 512 generated tokens; greedy ids the same as one card; these at 31976b51, before the waits' change, which leaves a group's logits the same bytes (Qwen3-8B Q8_0);
  - two stages of groups, GPU[2] and GPU[3] one group and GPU[6] and GPU[7] (PCI c3:00 and c6:00, another root complex) the other, lent for the check (190b1cb9): `llmx-split-check` bit-identical to the one group on Qwen3-0.6B Q8_0 and Qwen3-8B Q8_0 at ubatch 16 and 64 with 8 decode steps; `tools/server_mix_check.py` on Qwen3-8B Q8_0 over them, every request its reply alone together (8) and skewed (6, two clients leaving), and the CLI's (2); a group of GPU[2] and GPU[6] refused by name for its root complexes;
  - width-1 byte identity: Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3.5-0.8B Q8_0 in the load modes auto, mapped and direct, greedy ids, logits and perplexity, the same bytes on the CPU (9 runs) and on one MI50 (9 runs);
  - CTest 44 of 44 on an MI50, with `vulkan-lifetime`'s nine collective cases on the two cards; the suite on the CPU and on an MI50 with every component passing, the real-model baseline and `tensor-split` included; the Radeon VII on Windows (2dc3d132, a clean build), CTest 45 of 45, the collective cases skipping on its one device;
  - one timing round on one MI50, Qwen3-8B Q8_0 `bench --model --p 512 --n 128 --r 3`, interleaved main, branch, branch, main twice: pp512 main 841.1 / 835.1 / 836.1 / 835.3 and the branch 837.8 / 837.0 / 835.4 / 834.4 tok/s, tg128 main 77.56 / 77.41 / 77.34 / 77.20 and the branch 77.38 / 77.29 / 77.37 / 77.33 tok/s; the synthetic CPU bench on one thread, prefill main 5579 / 5569 / 5595 / 5606 and the branch 5642 / 5612 / 5583 / 5683 tok/s, decode main 5229 / 5193 / 5194 / 5192 and the branch 4944 / 5219 / 5186 / 5290; level within the runs' spread, the branch's 4944 a single low sample kept; other sessions' containers held other cards.
- **Open speed gate (first support):** each sum's waits went to the peers' submissions toward the same sum, so the second member to submit could not start a part until the first had finished it, and the members ran in turn; the waits now go to each member's next submission once every member has submitted (88decf0f).
  Two MI50s (GPU[2] and GPU[3]), RADV, default clocks, `bench --model --p 512 --n 128`: Qwen3-8B Q8_0 on a group pp512 1328 and tg128 78.8 tok/s, from 409 and 29.8 before the change (two runs each, interleaved), against one card's 837 and 77; Qwen3-32B Q8_0 on a group pp512 357.6 and tg128 29.7 tok/s (three runs), against the same two cards as a layer split 205.4 and 19.7, and the reference's ROCm tensor split on these cards (section 2.8) 570 and 33.6, so the group is at 63 and 88 percent of it.
  The server on Qwen3-32B Q8_0 over the same two cards, one build for both arms (b68e6a87, `llmx 0.1.0+gb68e6a87645f`, sha256 011227b0df5245c6), `serve --max-seqs 64 --ctx-size 65536` with and without `--tensor-width 2`, `tools/server_load.py --input-len 512 --output-len 128`, output tok/s, the arms interleaved group, split, split, group in one session:

  | users | group of 2 | layer split | reference tensor split | reference layer split |
  |---:|---:|---:|---:|---:|
  | 1 | 21.9 | 13.6 (e8995d76, section 2.8) | 26.7 | |
  | 4 | 44.3 | 34.4 (e8995d76, section 2.8) | 55.2 | |
  | 16 | 51.2 / 51.2 | 55.8 / 55.8 | 65.5 | |
  | 32 | 54.0 / 54.0 | 56.5 / 55.8 | 75.8 | 52.4 to 59.9 |
  | 64 | 51.1 / 53.3 | 57.1 / 56.9 | 87.5 | |

  The group's 1 and 4 users are from a run of the same build before the interleaved one; the reference's layer split at 32 users is the figure the MTP agent's session measured on these cards, beside its 57.5 to 57.7 for llmx's layer split on main, which the interleaved arm repeats.
  From 16 users both llmx arms are bound by their prompts, the group reading about 215 prompt tokens a second beside its decode rows against the 358 of one prompt alone, so the group is below the layer split there, by 4 to 9 percent, and below the reference's tensor split in every cell; the group's prompt rows beside decode rows are where the server cells are lost.
  Activity during the interleaved arms (10 s samples, host): other sessions' containers held other cards (two busy cards of a server on cores 4 to 7, and a reference server on GPU[5] using up to 1.7 cores on cores 12 to 15); nothing else ran on GPU[2] and GPU[3] or cores 8 to 11.
  Tried and not kept, Qwen3-8B Q8_0 on the same cards: every inbox on the first member, exported to the others, with the others submitting first, so that a dma-buf import's implicit wait would fall on work the importer needs anyway, pp512 1175 and tg128 76.8 against 1328 and 78.8 for the waits' change alone, so the implicit sync of section 4.3 costs nothing measurable once the waits are right; before the change, each member recording on a thread of its own (no gain), a submission after each sum's adds (31.5 against 34.1), and the inboxes in host memory (16.9), the host then blocking in each submission behind the queue's holds while the device caught up.
- **Left:** for step 4 or 5, the `server` and `chat` components on a group: under a tensor width they skip, their fixtures being refused by a group, so the HTTP routes, chat turns and reasoning splits over a group rest on `tools/server_mix_check.py` and the CPU groups of `server-passes-cpu` and `server-resume`; they are to run on the even tensor-split fixtures when a width is set rather than skip. Otherwise none for this step; the speed gate stays open under the first-support rule, its server cells lost to prompt rows beside decode rows and its prompt cells to be split between the collective and one card's kernels, which the recovery work after step 3 takes up before steps 4 to 8; the kernel route of section 8, item 3, waits on the user.
- **Gotchas:** the host rows of a group's logits are freed with the context, so a context must not be destroyed with a pass still copying into them; a wait a member takes from a peer belongs on its next submission after every member has submitted toward the sum, since a wait on a submission toward the same sum runs the members in turn. Each inbox holds a slot for every member though a member's own slot is never written, a waste of one slot in W; and `join` checks the PCI root before the exchange, so on a platform without sysfs every root reads empty and the check passes.

## Tensor groups on the CPU (2026-10-04, branch feat/tp-cpu, step 2 of TENSOR-SPLIT, lands by fast-forward)

- **Done:** `Collective` and `Backend::join`, and the CPU's; `Step::width`, `Step::partial` and `blocks::join`, and qwen3's and qwen35's full attention over a member's heads; the runtime's groups (`Placement::width`, a group named by its first member): the members' shards, their KV storages on the pool of the group's first member, `group_stage` and `end_stage`, the collectives and the members' logits slices in `ensure`, the members' host copies in `restore_host`; `group_budgets`, `placement_for` and `place_model` over groups; the loader's hook giving a member its shard; `--tensor-width` with its refusals and its page, the command line keeping its listed-once rule, so groups of CPU backends are the tools' (`llmx-split-check` and `llmx-model-logits` take a tensor width); new tiny HF fixtures whose every split falls whole at widths 2 and 4 (`tests/data/baseline_tensor_split.json`, `tools/gen_baseline.py tensor-split`, in the qwen35 venv) and the `tensor-split` component.
- **Gates** (model and loader tier), the branch at ae27d54e (`llmx 0.1.0+gae27d54ee0d8`, binary sha256 4a799a336d47225b) against step 1 at 46c96d18 (`llmx 0.1.0+g46c96d18a50b`, 6978211c26104f2c), which it lands on, both built with Vulkan the same way from detached worktrees in the development image, cores 8 to 11:
  - numerics of a group: the qwen3 fixtures, tied and untied, on groups of two and four CPU backends against HF within 3.4e-7 logits and 4e-8 NLL, against the F32 bounds of 2e-5 and 1e-5, every batched and per-token row at the goldens' positions, and each group passing the device-reference criterion against one backend, its rows and 64 greedy steps (`tensor-split`); one stage of width 2 against two stages of width 2 bit for bit, and within 1e-4 of one device (`placement`); the paused, held and steady server loads over 1 to 3 stages of groups of two at every P, and pauses, resumes, a donor taken back, a follow-up's fork and donors in host memory on groups, each reply its reply alone on one group (`server-passes-cpu`, `server-resume`); the collective's sums in member order at widths 2 to 4 (`backend-group`);
  - width-1 byte identity: Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3.5-0.8B Q8_0 in the load modes auto, mapped and direct, greedy ids, logits and perplexity, the same bytes on the CPU (9 runs) and on one MI50 (9 runs);
  - CTest 44 of 44 on an MI50, and the suite on the CPU and on an MI50 with every component passing, the real-model baseline and `tensor-split` included;
  - one timing round on one MI50, Qwen3-8B Q8_0 `bench --model --p 512 --n 128 --r 3`, interleaved base, branch, branch, base twice: pp512 base 840.8 / 836.1 / 836.0 / 835.2 and the branch 838.0 / 836.1 / 836.1 / 836.1 tok/s, tg128 base 77.47 / 77.30 / 77.40 / 77.25 and the branch 77.28 / 77.35 / 77.37 / 77.38 tok/s; the synthetic CPU bench on one thread, prefill base 5556 / 5586 / 5585 / 5070 and the branch 5574 / 5677 / 5593 / 5663 tok/s, decode base 5181 / 5252 / 5198 / 5067 and the branch 5211 / 5283 / 5208 / 5163; level within the runs' spread; activity not monitored beyond the other sessions' containers, which held other cards (GPU[5] and GPU[7]);
  - the suite's docs and dead-code components, `check_plan` and `pack` now called by the product, so their known findings are gone.
- **Gotchas:** a group runs no layer that keeps a state (step 5), no routed layer (step 8) and no embedded drafter (step 6), and holds the embedding, the head and every feed-forward block on its stages' groups; the collective is joined per context for the rows its arenas hold, where step 3's Vulkan inboxes may want it made once at load; the head's slices are copied into the logits rows by the first member, where step 3 adds a row stride to `matmul_logits`.
- Lands by fast-forward on the coordinator's review, the other developer away.

## serve --timing reads a busy stage as busy (2026-10-04, branch fix/timing-idle, lands by fast-forward)

- **Cause:** a backend made to time its work kept one query pool of 8192 timestamps, two a dispatch, and the scheduler reads each stage every 32 rounds; past the pool's 4096 dispatches nothing more was timed, so a large model's span counted only its first dispatches.
- **Done:** the test first (`backend-vulkan`: 6000 dispatches between two readings must all be timed, which main fails at 4096), then the fix: a reading interval takes as many query pools as its dispatches need, each reset as the interval first reaches it, and a reading empties them all.
- **Checks** (MI50s, cores 4 to 7): `backend-vulkan` fails at the test commit 1a066c14 ("device timing missed the dispatches past one query pool") and passes with the fix (6000 dispatches timed between two readings); CTest 43 of 43 with the fix, `vulkan-lifetime`'s query-pool failure and teardown cases among them. The same load as the finding, `serve --timing` with the fix, Qwen3-32B Q8_0 over two MI50s, 32 users of 512/128: `stage_idle` 0.049 and 0.018 (main read 0.72 and 0.70), `device_bound_rows_per_s` 288.7, the rows the load actually carried (main read 933), throughput 57.5 tok/s, as without `--timing`.
- **Gotchas:** a backend made to time its work and never read keeps adding pools, about 64 KB each (8192 timestamps of 8 bytes) per 4096 dispatches, without bound; the server reads every 32 rounds and bench after each phase, so neither grows far.
- **Landing:** the coordinator reviewed and approved it; the hosted run is green at its head; it lands by fast-forward.
## A lone prompt read by every stage (2026-10-04, branch perf/split-prompt-overlap, phase 4 of MULTI-DEVICE, lands by fast-forward)

- **Measured** on main c35b04b8, Qwen3-32B Q8_0 over two MI50s (GPU[6], GPU[7]), the server with 32-token replies (devlog 2026-10-04):
  - a trace of the scheduler's rounds (`exp/split-trace`) for one 2048-token prompt: four passes of 512 rows, each formed a few ms after the previous one's stage 0 finished, since a pass that wants no logits is ended without waiting for its device work; so slice k+1 runs on stage 0 while slice k runs on stage 1, and TTFT is 6.83 s, about five stage-times, against the CLI's own pipelined prompt (`bench --model` pp2048 on the split, 308 tok/s, 6.6 s). The overlap the phase planned for long prompts is already there.
  - what is left is the slice size for prompts not much longer than a ubatch: TTFT of a lone 512-token prompt 2455 ms at `--ubatch 512`, 1995 at 256, 1795 at 128, where a 512-row slice gives the second stage nothing to overlap; a lone 2048-token prompt 6070, 5785 and 5887 ms; four users of 512 tokens 15.3, 15.4 and 15.1 output tok/s; and at 32 users of 512 (devlog 15:45) `--ubatch 256` cost 4 percent, so the size has to follow the load, not be a smaller ubatch.
- **Done:** `prompt_slice` (`server/policy.hpp`), the one rule for a prompt slice's rows: a ubatch, but for a request alone, nothing else active, queued or paused after the round's admissions, about a 2 * stages-th of what it lacks, a ubatch at most and 128 rows at least, so a burst queued together never cuts, and company arriving mid-prompt first takes the lone request back to a whole ubatch, before any other prompt's slice and in a pass no other prompt shares, then whole ubatches; the scheduler's `form` and the `server-passes` round restated from it take it, and `server-passes` holds its values. Slices only move where a stretch is cut, so every row computes the bits it did.
- **First version, not kept:** cutting slices wherever the waiting prompt rows did not fill the stages (4bf4b2b8) took a lone 512-token prompt's TTFT from 2460 to 1797 ms on Qwen3-32B Q8_0 over two MI50s, but at 4 users of 512 it took TTFT from 4412 to 5219 ms and output from 34.5 to 33.9 tok/s, and at 32 users 57.3 to 55.8, since beside other requests the stages already have passes and the smaller slices only cost the tile; the rule now cuts a lone request's prompt only, and any lone prompt, since a 1024-token prompt's two 512-row slices also left a stage waiting (its TTFT 3784 ms on main).
- **Second version, not kept:** at e73abde8 a request was alone when no other was active, and an interleaved A/B on Qwen3-32B Q8_0 over two MI50s took a lone 512-token prompt's TTFT from 2449-2456 to 1797 ms but four users of 512 from 4390-4401 to 4741-4783 ms (+8.5%), which the review did not accept. Counting the queued and paused requests after the round's admissions (1041881b) left it at +7.8% (4405/4397 against 4741/4745 ms; Qwen3.6-27B Q8_0 +1.5%), and a trace of the scheduler's rounds (`exp/p4-trace`) showed why: the load opens its users within 1 to 5 ms, the first is formed alone and takes a cut slice before the others arrive, in the next round it is still in flight, so the second prompt takes a whole pass and the first reads its remainder in a short pass behind it, and stage 0 then sits idle about 330 ms while the host waits for that whole pass's logits, delaying every prompt after it. Hence the rule now in Done: back to a whole ubatch first, waiting for its own pass in flight, in a pass no other prompt shares.
- **Gates** at 94eb469c against main 235375a9 (`llmx 0.1.0+g94eb469c25b3`, 4f9bb3c650c39c45; `llmx 0.1.0+g235375a95215`, 97de9ff02189be6a), both built the same way from detached clones, cores 4 to 7, each timed arm started below 55 C, the host's load average 5.6 to 9.4 from other work through the A/B:
  - CTest 45 of 45 on an MI50, `server-passes` and `server-passes-cpu` with the burst and the arrival mid-prompt; the suite on the CPU and on an MI50 passes but raw-blocks (no numpy in the image) and, on the MI50, perf, whose synthetic prefill floor main misses alike on this host (529 to 545 tok/s on both, four runs interleaved, load 10.5); the Qwen3-0.6B and Qwen3.5-0.8B identity cells 14 of 14 on the CPU and on the device; `server_mix_check.py` on Qwen3-32B Q8_0 over two MI50s, 16 requests alone, together, skewed and against the CLI: every request matches; the linked dead-code check: 15 findings, all on the list.
  - interleaved A/B, two MI50s, fresh servers in the order main, branch, branch, main, 128-token replies, one round each, output tok/s and mean TTFT:

    | model, load | main tok/s | branch tok/s | main TTFT ms | branch TTFT ms |
    |---|---|---|---|---|
    | Qwen3-32B Q8_0, 512-token prompt, 1 user | 13.9, 13.8 | 14.9, 14.8 | 2458, 2463 | 1819, 1820 (-26%) |
    | Qwen3-32B Q8_0, 1024, 1 user | 13.3, 13.3 | 14.4, 14.3 | 2599, 2605 | 1915, 1925 (-26%) |
    | Qwen3-32B Q8_0, 512, 4 users | 34.3, 34.0 | 34.6, 34.1 | 4408, 4424 | 4425, 4433 (+0.3%) |
    | Qwen3-32B Q8_0, 128 to 2048, 4 users | 26.5, 26.4 | 25.6, 25.4 | 9085, 6466 | 7147, 7131 |
    | Qwen3-32B Q8_0, 512, 16 users | 55.2, 55.1 | 55.1, 56.0 | 12215, 12236 | 12218, 12241 (+0.1%) |
    | Qwen3.6-27B Q8_0, 512, 1 user | 17.2, 17.3 | 18.2, 18.2 | 1904, 1898 | 1520, 1514 (-20%) |
    | Qwen3.6-27B Q8_0, 1024, 1 user | 16.6, 16.6 | 17.3, 17.2 | 2172, 2178 | 1856, 1871 (-14%) |
    | Qwen3.6-27B Q8_0, 512, 4 users | 43.2, 43.2 | 42.9, 42.9 | 4818, 4812 | 4907, 4914 (+1.9%) |
    | Qwen3.6-27B Q8_0, 128 to 2048, 4 users | 33.0, 33.1 | 32.7, 33.0 | 6591, 6578 | 6885, 6450 |
    | Qwen3.6-27B Q8_0, 512, 16 users | 67.7, 70.6 | 68.9, 69.2 | 11643, 11625 | 11730, 11706 (+0.7%) |

    Traces of the 4-user wave at 94eb469c (`exp/p4-trace2`, five fresh servers) show the cut on every run: on Qwen3-32B Q8_0 the passes then go 128, 384 (the first prompt alone), 512, 512, 512, where main goes 512 four times, and TTFT is level. On Qwen3.6-27B Q8_0 a prompt's slices also end at its checkpoint, the last whole block before its end (448 of 512), so the first prompt goes 128 and 320 where main reads 448 at once, and every later pass is main's; the +90 ms is that one extra pass, about its fixed cost, which company pays when the first of a burst is formed before the others arrive and which no rule avoids without waiting for arrivals. The 128 to 2048 rows follow which prompt reaches the server first, a race of about a millisecond that the shorter prompts usually win: traced twice an arm on Qwen3-32B Q8_0, main 24.3 and 24.0 tok/s at 7784 and 7785 ms, the branch 24.4 and 24.3 at 7802 and 8712 ms, the slower run being the one where the 776-token prompt arrived first.
  - at e73abde8, the second version, against main 9091ee2e: the load table at 1, 4, 16, 32 and 64 users, one round a level, the main arms from the afternoon's run on the same cards (p4gate-4bf4b2b8), output tok/s: Qwen3-32B Q8_0, 512 tokens: main 13.9, 34.5, 56.5, 57.3, 58.8 and branch 14.9, 34.8, 56.4, 57.4, 56.4; 1024: main 11.9, 25.2, 34.5, 34.9, 34.0 and branch 12.1, 25.1, 34.8, 35.0, 34.9; 128 to 2048: main 12.5, 24.5, 32.5, 31.6, 28.7 and branch 12.8, 23.1, 33.1, 31.5, 29.0. Qwen3.6-27B Q8_0, 512: main 17.4, 43.4, 70.4, 72.4, 76.4 and branch 18.4, 43.1, 71.1, 73.0, 74.7; 1024: main 15.0, 32.1, 44.7, 45.5, 44.7 and branch 15.2, 32.0, 45.0, 45.7, 44.3; 128 to 2048: main 15.7, 31.1, 42.2, 40.0, 35.9 and branch 15.6, 30.4, 42.5, 40.5, 36.4; and at 64 users of 512 on Qwen3-32B Q8_0, interleaved, 60.7 and 59.7 tok/s against 60.9 and 61.0.
  - the reference's layer split (pipeline parallelism in its log, `-b 4096 -ub 1024`) on the same cards: with a slot's context of 1024 (`-c 65536 -np 64`), 512-token prompts, 15.2, 34.2, 50.3, 62.2, 65.1 tok/s on the 32B and 17.2, 26.4, 39.8, 43.3, 39.0 on the 27B; 1024-token and longer prompts do not fit those slots, and with the KV unified over the slots (`-kvu`) it slows under load and fails at 64 users (32B 512: 14.7, 31.5, 30.1, 23.3, 24.4; 1024: 12.2, 20.3, 15.3, 8.9, none), so those rows say little. Its lone-prompt TTFT, 1537 ms (32B) and 1437 ms (27B) at 512, is near the branch's.
- **Rebased** onto main a2f32b8f, the one commit 8712ccb4 (`llmx 0.1.0+g8712ccb46952`, 436b0c5e8721885f; main `llmx 0.1.0+ga2f32b8fbed4`, 067d6e458f3e9add), no conflict in code: CTest 45 of 45 on an MI50; the suite on the CPU and on an MI50 passes but raw-blocks; the identity cells 14 of 14 on the CPU and on the device; `server_mix_check.py` as above, every request matching; the linked dead-code check: 16 findings, all on the list.
- **Accepted cost** (the coordinator, 2026-10-05, under the tradeoff rule): Qwen3.6-27B Q8_0, 512-token prompts at 4 users, TTFT 4818/4812 to 4907/4914 ms (+1.9%, about 90 ms), against a 14 to 26 percent lone-prompt TTFT gain on both models with output tok/s level everywhere and the Qwen3-32B rows level. Its traced cause: the first request of a burst is formed before the others arrive and reads 128 then 320 rows to its checkpoint where main reads 448 at once, one extra pass whose fixed cost the requests after it pay; no rule removes it without waiting for arrivals.
- **Landing:** the coordinator reviewed and accepted it; the hosted run is green at 9a321f88 (the CPU ubuntu-24.04 job rerun once after an unrelated `disk-store` timeout); it landed by fast-forward.
- **Gotchas:** per-storage progress and dependent-chunk cancellation, which the phase planned before overlapping a prompt's chunks, are not needed for this: each slice is still a pass of its own with the sequence in at most one pass, and the next slice forms only once the previous one has ended.

## Tensor shards (2026-10-04, branch feat/tp-shard, step 1 of TENSOR-SPLIT, lands by fast-forward)

- **Done:** `Role::shard` (an `Axis` and `ShardSection`s of tiles of units, a unit replicated where the section allows, as KV heads are) and the declarations: the head by vocabulary rows, q and gated q by heads, k and v by KV heads, `attn_output` by the heads' columns, the dense block by `blocks::shard_swiglu`, and a linear-attention layer by K heads, `attn_qkv` and the conv's channels as each member's q and k rows and the V heads of its K heads from every tile, z, alpha, beta, the decay, the time step and `ssm_out`'s columns by those V heads; `shard::spans`, `check_plan` (refusals by name: indivisible heads, KV heads, K heads and vocabulary rows, columns off whole quant blocks, routed layers and an embedded drafter), `runs`, `bytes`, `pack`, `shape`, `kv_heads` and `state`; `footprint(weights, plan, options, width, member)`; `Upload::runs` and `detail::stream` sending a member only its runs.
- **Gates** (model and loader tier), the branch at 387c31a9 (`llmx 0.1.0+g387c31a97002`, binary sha256 ff1355cad0290bb8) against main ed78eaa6 (`llmx 0.1.0+ged78eaa65a26`, ff0f66ec2755a2ff), both built with Vulkan the same way from detached worktrees in the development image, cores 8 to 11:
  - width-1 byte identity: Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3.5-0.8B Q8_0, each in the load modes auto, mapped and direct, greedy ids of 48 tokens, the top-10 logits after a prompt and the perplexity of 4 windows of 128 tokens, the same bytes on the CPU (9 runs) and on one MI50 (9 runs);
  - CTest 44 of 44 on an MI50 (`shard` 12.5 s, `model-validation` with its 7 new refusals at the end of the list, `load-progress`), the suite on the CPU and on an MI50 (`--device vulkan:0`) with every component passing, the real-model baseline included (the qwen3 gate files and Qwen3.5-0.8B Q8_0 and Q4_K_M);
  - one timing round on one MI50, Qwen3-8B Q8_0 `bench --model --p 512 --n 128 --r 3`, the arms interleaved main, branch, branch, main twice: pp512 main 820.2 / 816.0 / 819.4 / 818.1 and the branch 819.1 / 818.8 / 818.5 / 818.5 tok/s, tg128 main 76.17 / 77.13 / 77.07 / 77.09 and the branch 77.14 / 77.14 / 77.06 / 77.09 tok/s; the synthetic CPU bench on one thread, main prefill 5532 / 4899 and decode 5128 / 5134 tok/s, the branch 5678 / 5642 and 5280 / 5246; level within the runs' spread; machine activity not monitored beyond the other developers' containers, which held other cards;
  - the suite's docs and dead-code components, with `check_plan` and `pack` listed as known findings naming step 2, which calls them.
- **Gotchas:** width 1 is byte-identical, since no declaration changes what a role reads or how it is adopted; `footprint` keeps one device's arena, handoff rows and mark rows for a member, an upper bound; the block-boundary rule applies span by span, so a tiled split is checked on each tile.
- Lands by fast-forward on the coordinator's review, the other developer away.

## Host-relayed events measured, and the kernel route proposed (2026-10-04, branch tools/tp-relay, tensor split transport research, lands by fast-forward)

- **Goal:** the transport routes of `docs/TENSOR-SPLIT.md`, section 8, item 3, after the coordinator's direction (the kernel route proposed, not tried on the test machine, which hosts production; the host-relayed events and the cheaper sync-file chain measured).
- **Done:** `llmx-vk-handoff exchange` with host inboxes also times host-relayed events: every epoch in one command buffer a member, its partials into the host inboxes, a barrier to the host and an event the host polls, and the host setting each member's event its sum waits on; and the same chain with each member setting its own event, with and without the barrier, which isolates the cost.
  Two MI50s of one root complex (PCI 83:00 and 86:00), RADV Mesa 25.0.7, default clocks, 64 epochs a chain, median of 5 chains, three runs: at 20 KB host-relayed events 135.1 to 138.2 us an epoch, sync files 139.3 to 146.6, the members' own events 137.3 to 140.5 and without the barrier 138.7, over a floor of 19.5 to 19.8; level at 160 KB (181.6 to 184.2 against 163.1 to 182.4), 1.25 MB and 10 MB; every sum correct.
  The plan records the result: the command processor's event set and wait costs about 118 us an epoch, which closes the host-relayed route and, at width 2, the cheaper sync-file chain; and it proposes the smallest kernel change, an import inheriting `AMDGPU_GEM_CREATE_EXPLICIT_SYNC` from an amdgpu exporter, with its rationale and risks, untried until the user decides how to test it.
- **Checks:** a tools and docs change: the docs and dead-code components; the tool builds in the Vulkan configuration; the hosted run.
- **Gotchas:** the chains without the host do not order the members, so their sums are not counted; the members' arrival spread stays about 55 us at decode sizes on every chain.

## Q8_0 decode by two-wide 16-bit dots (2026-10-04, branch perf/q8-decode-5to8, lands by fast-forward)

- **Cause found:** the builds of several columns were bound by their instructions, not their weight reads. Each product of a weight word and four 16-bit activations took two four-wide 8-bit dots over the activation's high and low bytes, a correction by the weights' sum and the shifts and masks that split every activation word, per column; the 8-column build issued 2337 vector instructions where the 4-column build issued 964 (RADV, MI50).
- **Done:** `dot16.glsl` widens each weight word once per load to two pairs of signed 16-bit values and takes two two-wide 16-bit dots (`v_dot2_i32_i16`) per word for every column, each dot taking the sum so far as its accumulator; the integer-dot tile's own copy of the same widening and dots moved there, so it has one owner. The integer sums are exact either way, so every column computes the bits it did. Vector instructions per build (RADV, MI50): 1 column 240 to 226, 2 columns 360 to 288, 4 columns 964 to 724, 8 columns 2337 to 1474, 16 columns 5400 to 3440; the 1-column build takes 40 VGPRs against 32 (6 subgroups a SIMD against 8), the others unchanged.
- **Tried and not kept** (`exp/q8-rows` on Gitea, never to land), each on the two-wide dots, Qwen3.6-27B-MTP Q8_0, one MI50, the 8-column build's sampled device time at 5 / 8 columns against 335 / 400 ms: 2 rows a subgroup 551 / 667 and 8 rows 363 / 415 (one subgroup a SIMD); two steps of weights loaded ahead 328 / 401; the subgroup reduction instead of the transposed one 350 / 410; a column group's activations staged once a workgroup in shared memory 345 / 396; workgroups of 512 355 / 411; 5 to 8 columns as the 4-column build twice over the same rows on adjacent workgroups 439 / 460. For 9 to 16 columns the 8-column build over two workgroups gave 647 / 761 ms at 9 / 16 columns against 659 / 832 for the 16-column build, too small at 9 to carry a change; for 17 to 32 it lost (1506 against 1335 ms at 32). So the step from 4 to 5 columns stays, at about 30 percent of the kernel's time, and every build of several columns is cheaper.
- **Gates**, main e8995d76 (`llmx 0.1.0+ge8995d7682cf`, binary sha256 976c1a43feb6996f) against the change (723b4089, `llmx 0.1.0+g723b40890105`, d5478f1a9a82f715), both built the same way from detached clones, cores 4 to 7, each timed arm started below 55 C at default clocks:
  - batch invariance: `backend-vulkan` with `--isa` passes on an MI50 (RADV), every decode column of every build bit for bit the column alone, and its instruction screens hold the six Q8_0 decode builds to the counts their shape and forms give; CTest 43 of 43 on an MI50; the suite on the CPU and on an MI50 with `--require-tools` passes but raw-blocks, which fails only for want of numpy in the image (CI installs it).
  - byte identity: 28 of 28 Qwen3 cells (greedy, seeded, logits, the excerpt's last rows, perplexity batched and per token, chat) the same on one MI50 for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0, Qwen3-8B Q4_K_M (the integer-dot tile) and Qwen3.6-27B-MTP Q8_0, 14 of 14 over two MI50s for the 27B and Qwen3-30B-A3B Q8_0 (the routed and grouped builds), and the suite's Qwen3-0.6B and Qwen3.5-0.8B cells 14 of 14 on the CPU and on the device.
  - HF: the bits are main's, so the errors are main's; the device suite's Q8_0 MoE fixture, whose decode runs this kernel, is at most 0.000163 logits from HF against its 0.127 bound.
  - Radeon VII (AMD driver, Windows): `backend-vulkan --isa` passes on both arms; its profile takes neither this kernel nor the integer-dot tile (its passes run `matmul_row_q8w` and the float tiles), so the change reaches no kernel there: Qwen3-8B Q8_0 `--seqs` 1 to 9 and pp512 / tg128 level with main (tg128 41.30 and 40.93 main, 41.17 and 41.01 the change).
  - timing on one MI50, `bench --model` Qwen3.6-27B-MTP Q8_0, 64-token prompts and 32 generated tokens a sequence, two runs an arm, the change then main at each count, tok/s together and the Q8_0 decode kernel's sampled device time:

    | sequences | main tok/s | change tok/s | change | main kernel ms | change kernel ms | build |
    |---|---|---|---|---|---|---|
    | 1 | 22.38 | 21.30 | -4.8% | 221.1 | 219.4 | 1 column |
    | 2 | 36.03 | 35.29 | -2.1% | 247.2 | 248.0 | 2 columns |
    | 3 | 47.78 | 52.80 | +10.5% | 280.9 | 249.0 | 4 columns |
    | 4 | 58.54 | 69.28 | +18.3% | 296.7 | 252.2 | 4 columns |
    | 5 | 60.24 | 66.31 | +10.1% | 379.5 | 329.9 | 8 columns |
    | 6 | 65.40 | 74.02 | +13.2% | 412.2 | 351.8 | 8 columns |
    | 7 | 69.82 | 79.48 | +13.8% | 440.0 | 376.9 | 8 columns |
    | 8 | 72.70 | 85.34 | +17.4% | 485.0 | 403.2 | 8 columns |
    | 9 | 52.94 | 64.08 | +21.0% | 827.9 | 658.7 | 16 columns |

    At 1 and 2 sequences the kernel's time is the same in both arms and the totals differ within their runs' spread (+-0.46 at 1); the decode-only runs below settle one sequence. Per column, the 8-column build now costs 66.0 ms at 5 columns, 58.6 at 6, 53.8 at 7 and 50.4 at 8 against the 4-column build's 63.1 at 4, where main's 4-column build cost 74.2.
    Qwen3-8B Q8_0 pp512 / tg128 on an MI50 (GPU[7]), two rounds: main 830.8 / 74.98 and 825.7 / 74.25, the change 833.3 / 77.13 and 830.2 / 77.25 (+0.4 / +3.4 percent); Qwen3-8B Q4_K_M, whose prompts run the integer-dot tile: main 775.4 / 87.88 and 768.9 / 87.69, the change 775.8 / 87.88 and 775.8 / 87.68.
  - MTP and serving: generate greedy, 256 tokens with the end ignored, on the step-4 prompts, on one MI50 (GPU[1]), in the order main, change, change, main, the mean of each arm's two runs in tok/s; the ids are the same in every arm and at every depth of a prompt:

    | prompt | main off | change off | main depth 3 | change depth 3 | main depth 4 | change depth 4 |
    |---|---|---|---|---|---|---|
    | copy | 21.03 | 20.94 (-0.4%) | 40.41 | 44.85 (+11.0%) | 38.00 | 42.12 (+10.8%) |
    | explain | 20.99 | 21.21 (+1.0%) | 37.63 | 41.25 (+9.6%) | 33.89 | 37.23 (+9.9%) |
    | code | 21.20 | 21.64 (+2.1%) | 43.34 | 48.51 (+11.9%) | 40.12 | 43.61 (+8.7%) |

    Depth 4 still gives fewer tokens a second than depth 3, since the step at 5 columns stays (USAGE says so).
    The server, `tools/server_load.py` on the 27B with `--max-seqs 8 --ctx-size 8192`, the eight fixed prompts, 256 tokens a reply, three rounds a level, output tok/s at 4, 5, 6, 7 and 8 users: drafts off, main 58.3, 52.4, 56.8, 61.2, 64.0 and the change 63.5, 58.9, 66.2, 71.8, 75.3 (+9 to +18 percent); `--drafter embedded --draft-max 3`, main 50.9, 51.2, 54.9, 58.3, 62.3 and the change 61.5, 61.5, 65.8, 71.4, 78.2 (+20 to +26 percent). At 4 users drafting gives less than drafts off in both arms (50.9 against 58.3 on main), which the pass price does not yet avoid; it is recorded, not part of this change.
    A one-second monitor read a load average of up to 10.7 from other work on the host and the card at up to 105 C during the server levels; each arm started below 55 C. The first gate run's generate and drafting server arms did not run (the script's cooling loop overwrote the depth variable), and were run again in full; its drafts-off server levels for main, 61.0, 53.1, 56.4, 60.9 and 63.7, agree with the rerun's.
- **Landing:** the coordinator reviewed the shaders, the device needs and the prefill round; XDEV's review was requested (devlog 2026-10-04 11:03), and with XDEV away it lands on the coordinator's review at the user's word; the hosted run is green at its head; it lands by fast-forward.
- **Prefill** (the coordinator's review, since the integer-dot tile now takes its widening from `dot16.glsl`): `bench --model` on one MI50 (GPU[1]), three runs an arm, in the order main, change, change, main twice, each arm started below 55 C, tok/s, the mean of each arm's four:

  | model | main pp512 | change pp512 | main pp2048 | change pp2048 |
  |---|---|---|---|---|
  | Qwen3-8B Q8_0 | 841.19 | 846.28 (+0.6%) | 771.12 | 775.26 (+0.5%) |
  | Qwen3-8B Q4_K_M | 782.52 | 788.27 (+0.7%) | 719.51 | 719.98 (+0.1%) |
  | Qwen3.6-27B-MTP Q8_0 | 258.42 | 260.08 (+0.6%) | 249.79 | 251.27 (+0.6%) |

  The change is never the slower arm of a pair; every cell is well inside the layout band. A one-second monitor read the host's load average between 7.6 and 51 from other work (highest during the 8B pp2048 cells) and the card at up to 101 C.
- **Device needs** (the coordinator's review): the kernel's 16-bit integer types are a need every Vulkan kernel declares (`missing_device_need`, "has no 16-bit integer arithmetic"), which `vulkan-buffer` refuses by name. Its 16-bit dots, like the packed 8-bit dots it took before and the tile's 16-bit dots, come under the integer dot product's one feature (`shaderIntegerDotProduct`, which grants `DotProductInputAll`); that is not a need but what chooses the kernel: only a profile that prefers the integer dot selects this kernel and the integer-dot tile, and `profile_for` keeps that only where the device has the extension. `vulkan-buffer` now checks the gate on a device's names and caps alone: the MI50 row with the integer dot prefers it, without it takes neither the integer dot nor MXFP4's; with the gate removed the check fails.
- **Gotchas:** a driver that lowered `dotEXT` on 16-bit pairs to anything but a native dot would lose; `--isa` shows `v_dot2_i32_i16` on RADV.

## Drafter files beside a model (2026-10-03, branch feat/spec-drafters, step 6 of SPECULATIVE)

- **Sidecar files on the MI50 host** (GGUF headers, 2026-10-03): the two DFlash drafters (Qwen3.8-27B-DFlash2 Q8_0, 5 blocks, n_embd 5120, block 8, taps 6 to 62; Qwen3.6-35B-A3B-DFlash Q8_0, 6 blocks, n_embd 2048, block 16, taps 2 to 38), both with the qwen35 tokenizer, which no proposer runs until step 7; the qwen4exp MTP heads, an architecture llmx does not run; no standalone qwen35 MTP file, every qwen35 MTP block being embedded; draft models of a matching tokenizer, Qwen3-0.6B for the Qwen3 models and Qwen3.5-0.8B, 2B and 4B for the Qwen3.6 and 3.8 models.
- **Done:** drafting fitted after the no-drafter fit (`fitted_kv`: the budget and the checkpoints without the embedded drafter or a mark, then the drafter and one mark beside them or a refusal with the numbers, then the other marks), which the embedded drafter had not kept, it being counted inside the KV bisection; the pairing, the loader's resolution, the draft-model proposer, `--drafter PATH` on `generate`, `chat`, `serve` and `bench --model`, the pack tool, `llmx-decode-probe`'s `drafter_file`, the `drafters` suite component, and their docs; CPU: CTest 39 of 39, the suite's drafters, qwen35, decode-probe, cli, docs and dead-code.
- **Gates so far**, MI50s (GPU[1] alone, GPU[6] and GPU[7] for two), cores 4 to 7, each arm started below 55 C at default clocks:
  - at a69501f2, rebased on main 6a365411: CTest 43 of 43, the device suite's qwen35, drafters, decode-probe and server; `server_mix_check.py` on Qwen3.6-27B-MTP Q8_0 split by the tool, the blocks beside it, `--ctx-size 8192 --max-seqs 4`: every request equal alone, together, skewed and to the CLI, 722 drafts fed and 508 kept;
  - at 239b74d5: Qwen3.6-27B-MTP Q8_0 split by `llmx-drafter-pack` (28.6 GB model and 462 MB of blocks, written to the root pool, /zpool1 being 99 percent full): `llmx-decode-probe`'s drafts and every draft row's logits from beside the model equal to embedded after three prompts; `generate`, 256 tokens after the wiki test set's first 1400 bytes, ids the same off, embedded and beside, greedy and seeded, at 21.8, 36.6 and 36.8 tok/s greedy, the same drafts kept (69/91, 53/90, 42/90);
  - draft models, the same prompt, `generate` tok/s, ids on equal to off in every arm, against the reference's `llama-server --spec-type draft-simple` at the same depth and one request (the reference measured at 13633e8b's session, an hour before, on the same cards):

    | target and draft model | off | depth 3, greedy | depth 8, greedy | depth 3, seeded | depth 8, seeded | reference off | reference depth 3 | reference depth 8 |
    |---|---|---|---|---|---|---|---|---|
    | Qwen3-8B Q8_0 + Qwen3-0.6B Q8_0, one MI50 | 70.0 | 115.6 | 79.3 | 75.4 | 34.3 | 68.4 | 107.4 | 74.6 |
    | Qwen3.6-27B Q8_0 + Qwen3.5-0.8B Q4_K_M, one MI50 | 21.8 | 29.1 | 14.7 | 30.8 | 14.0 | 21.5 | 19.6 | 12.1 |
    | Qwen3-32B Q8_0 + Qwen3-0.6B Q8_0, two MI50s | 18.5 | 32.0 | 17.1 | 24.3 | 10.3 | 17.9 | 32.3 | 18.95 |

    A draft model's catch-up went through the prompt path at first, one or two rows taking 14 to 17 ms against 3 ms a decode step of Qwen3-0.6B, which held Qwen3-8B with the 0.6B at 82 tok/s; through the decode kernels it gives 116. The first reference run of these pairs passed `-md` without `--spec-type draft-simple`, drafted nothing and is not used.
  - serve starts with the fit rule: on one MI50 the embedded drafter and the blocks beside the model are refused at `--ctx-size 8192` (8192 tokens and 8 checkpoints leave no room) and at the model's context (22592 tokens and 3 checkpoints); over two MI50s at the model's context it starts with 262144 tokens and 8 checkpoints, as without drafts.
- **Fit order** (asked in the devlog 2026-10-04 00:37, option B, which XDEV agreed at 00:39 with its invariant): the no-drafter fit, its automatic checkpoints included, fixes the budget, which drafting never changes; the drafter and a mark then take room from the automatic checkpoints, the most of which that still fit stay, never more than without drafts, and a `--state-checkpoints` given by number is held or the placement refused.
  - at eed6edff, with the fit order: CTest 43 of 43 and the device suite's four components; `serve` on one MI50 at `--ctx-size 8192 --max-seqs 8` starts with the embedded drafter and with the blocks beside the model over 8192 KV tokens and 4 automatic checkpoints, where it holds 8 without drafts; at the model's context it is refused by its text, 22592 tokens leaving no room even with no checkpoint, so a one-MI50 server with drafts takes a `--ctx-size`; over two MI50s at the model's context it starts with 262144 tokens and 8 checkpoints with the drafter as without; `server_mix_check.py` with the blocks beside the model, `--ctx-size 8192 --max-seqs 4`: every request equal, 766 drafts fed, 557 kept. One start with the blocks beside the model, five seconds after another server on the card had stopped, read the card before its memory had come back and was refused at 6208 tokens; started again on the settled card it took 8192 and 4 checkpoints.
- At 6b9d37be, on main bf0d1a93: CTest 43 of 43 and the device suite's four components on an MI50; the one-MI50 and two-MI50 starts as at eed6edff; `server_mix_check.py` with the blocks beside the model all equal, 777 drafts fed and 566 kept.
- **Finding, the free-memory settle, resolved on 2026-10-05 by `fix/fit-settle-whole`:** a server started about 5 s after another had stopped on the same MI50 read the card before its memory had come back and fitted a smaller budget. Qwen3.6-27B-MTP Q8_0 with its MTP blocks beside it at `--ctx-size 8192 --max-seqs 8` was refused at a budget of 6208 tokens (at eed6edff) and of 7616 tokens (at 6b9d37be), where on a settled card the same start takes 8192 tokens and 4 checkpoints. The first reading held the budget alone, so the fit read no further.
- **Landing gates**, main bf0d1a93 against the landing code (5b65086d), both built the same way from detached clones, on GPU[1], GPU[6] and GPU[7], cores 4 to 7: CTest 43 of 43; byte identity with no drafter, Qwen3-0.6B, Qwen3-8B and Qwen3.6-27B-MTP Q8_0, 21 of 21 cells the same on the CPU and on one MI50, and the 27B over two MI50s 7 of 7.
  The fitted serve budgets, Qwen3.8-27B Q8_0 at default flags, each start after the cards had settled: over two MI50s 262144 KV tokens and 16 checkpoints in both arms without a drafter and in both with `--drafter embedded`; on one MI50 without a drafter 8192 KV tokens and 1 checkpoint taking 2432 tokens in both; on one MI50 with `--drafter embedded` both arms refused, main with no room for one KV block, the drafter having been counted inside the budget, and this change by its text, the 8192-token budget leaving no room for the drafter even with no checkpoint.
  One timing round, `bench --model` Qwen3-8B Q8_0 on one MI50, pp512 / tg128 tok/s in the order main, change, change, main: 838.9 / 75.02, 832.6 / 75.24, 829.2 / 74.89, 829.2 / 74.95, the change -0.4 percent at pp512 and +0.1 percent at tg128 on the means; a one-second monitor read a load average of 6.3 at most and the card at up to 87 C while it ran, each arm started below 55 C.
- **Landing:** the coordinator reviewed the squash and approved the fit order B; the hosted run is green at its head; it lands by fast-forward.

## A pass's stages in one file (2026-10-03, branch refactor/runtime-split, step 0b of TENSOR-SPLIT, move only, lands by fast-forward)

- **Done:** `src/model/passes.hpp` holds how a pass runs: `Model::forward` and the pass API (`reserve_passes`, `begin_pass`, `run_pass_stage`, `pass_logits`, `end_pass`, `abort_pass`), and the private steps of a pass, `begin`, `run_stage`, `draft_context`, `finish`, `roll_back`, `alloc_arena`, `ensure`, `send`, `receive`, `cross`, `ffn_split`, `part` and `mixer_part`, each body and its comments moved unchanged and defined out of the class in the class's order; `Model` declares them in three groups where the first of each stood, and `runtime.hpp` includes the file after `history.hpp`. The small helpers a pass shares with the history operations and the entry points (`whole_blocks`, `in_flight`, `release`, `handoffs`, `scoped`, `slot`, `row`, `streams`, `retire`) stay in the class. `runtime.hpp` goes from 1400 lines to 976. `docs/src/model-passes.md` takes the description of `forward`, the pass API and the crossings from the runtime's page.
- **Gates** (`bdb1e1e3`, this commit before this block's gate lines): the 430 moved non-blank lines are the removed ones, in order, but for the indent, `inline` and `Model::` before each name and `begin`'s default argument, which stays on its declaration; the 888 non-blank lines left in `runtime.hpp` are main's but for the added declarations and the include. On Windows: `build.bat` and a fresh CMake build without a warning in a changed file, CTest 39 of 39, `docs` and `dead-code` [ok]. On the Linux MI50 machine against main `6a365411`, each arm built the same way from its own shallow clone detached at its commit, Vulkan on (main `llmx 0.1.0+g6a365411a11e`, sha256 `9cf8cead...`; branch `llmx 0.1.0+gbdb1e1e353c9`, sha256 `d7340128...`): Qwen3-0.6B, Qwen3-8B and Qwen3.5-0.8B Q8_0 give main's stdout in all 63 cells (generate greedy and seeded, logits `--top 20` and `--last 4`, perplexity batched and `--per-token`, two chat turns, each on the CPU, one MI50 and two MI50s split), the timing lines of generate aside, and main's stderr in all 63; `llmx-split-check` one MI50 against two (excerpt, 8 steps, ubatch 64) bit-identical on all three models with both arms, its stdout byte-equal between them; CTest 43 of 43 with Vulkan and 38 of 38 without; the suite's split, server, qwen35, f32 and moe components on the build without Vulkan all PASS with `--require-tools`, the server's real-model pass skipped without the Qwen3-0.6B fixture in the container's cache. The hosted run is not part of these gates yet.
- **Timing** against main on one MI50 at default clocks, `bench --model Qwen3-8B-Q8_0 --p 512 --n 128 --r 3 --device vulkan:0`, each arm's three repetitions a run, tok/s in run order, a perturbed-layout control (main with one unused function appended to `src/model/history.hpp`, committed, `llmx 0.1.0+gbe133491e294`) in a second block; the monitor of this round sampled every 5 s only the host load average (7.4 to 11.9 on the 16-thread machine) and GPU[2] and GPU[3] use and power (GPU[3] idle throughout), so per-process CPU, system CPU and disk activity are unavailable for its 16 runs and the load is not attributed:

  | block | cell | main | change or control | change in the mean, percent |
  |---|---|---|---|---:|
  | main, branch, branch, main, twice | pp512 | 833.93, 831.10, 829.96, 828.80 | 832.36, 830.96, 829.43, 830.03 | -0.03 |
  | main, branch, branch, main, twice | tg128 | 74.57, 74.88, 74.98, 74.86 | 74.93, 74.96, 74.92, 74.93 | +0.15 |
  | main, control, control, main, twice | pp512 | 829.79, 829.58, 829.52, 829.27 | 829.66, 829.58, 830.75, 830.05 | +0.06 |
  | main, control, control, main, twice | tg128 | 74.98, 74.88, 74.68, 74.93 | 74.87, 75.05, 75.03, 74.93 | +0.13 |

  pp512 drifts down about 0.5 percent over the session in every arm. That control perturbs `history.hpp`, not the files the change edits, so it does not give the band at the change's location.
- **Timing repeat at the matched location** (after review): the same command, one MI50, default clocks, main, a control of main with one unused non-inline function appended to the end of `src/model/runtime.hpp` (committed, `llmx 0.1.0+g95f18bba9173`, the function present in the binary) and the branch, in the order main, control, branch, branch, control, main, twice, every run kept. A monitor sampled with a nominal 1 s delay after each collection (observed interval median 1.42 s, at most 1.47 s), the same for every arm: per-process CPU (`top`), system CPU (`/proc/stat`), disk (`/proc/diskstats`) and GPU[2] and GPU[3] use and power (`rocm-smi`); the activity flags were written before measuring: F-gpu3 (GPU[3] above 0 percent), F-cpu-host (all 16 threads above 50 percent), F-cpu-sib (cores 0-3 above 50 percent), F-cpu-mine (cores 8-11 above 50 percent), F-proc (a process other than the bench and the monitor at 100 percent or more), F-disk (above 100 MB/s).

  | run | arm | pp512 | tg128 | flags |
  |---:|---|---:|---:|---|
  | 1 | main | 835.77 | 74.98 | F-disk (2354 MB/s), F-proc (`llmx-spec-test`) |
  | 2 | control | 830.37 | 74.95 | F-disk (625 MB/s) |
  | 3 | branch | 828.55 | 74.92 | F-cpu-mine |
  | 4 | branch | 829.17 | 74.90 | F-cpu-mine |
  | 5 | control | 827.62 | 74.83 | none |
  | 6 | main | 830.20 | 74.97 | F-cpu-mine, F-proc (`llmx-mxfp4-vulk`, as `top` cuts the name) |
  | 7 | main | 829.98 | 74.91 | F-cpu-mine |
  | 8 | control | 830.47 | 75.01 | F-cpu-mine |
  | 9 | branch | 830.31 | 75.07 | none |
  | 10 | branch | 829.07 | 75.02 | F-cpu-mine |
  | 11 | control | 829.28 | 74.98 | F-cpu-mine |
  | 12 | main | 828.61 | 75.02 | F-cpu-mine |

  Against main's mean, the control is -0.21 percent on pp512 and -0.04 on tg128, the branch -0.22 and +0.01, so the branch sits inside the band the matched-location control shows. No run saw F-gpu3, F-cpu-host or F-cpu-sib (all 16 threads at most 23.9 percent, cores 0-3 at most 16.2). Cores 8-11 passed 50 percent in eight runs, three of main's, two of the control's and three of the branch's; the bench's own process took 7 to 27 percent CPU in `top`, so most of that load was other processes, and `top` showed test binaries running elsewhere (`llmx-spec-test`, `llmx-backend-vu`, `python3`, names as `top` cuts them) but records no core affinity, so which process ran on cores 8-11 is not known. Run 1, main's fastest, also read the disk at up to 2354 MB/s.
- **Left:** the hosted run; landing by fast-forward.
- **Gotchas:** a branch that changed one of these bodies in `runtime.hpp` moves its change to `passes.hpp`, where the body sits dedented by one level, `inline` and `Model::` before its name.

## The edited-turn gap after message boundaries: what it is made of (2026-10-04, finding, docs only, lands by fast-forward)

- **Question:** after message boundaries (step 2b part b), an edit at turn 2 of the six-user, twenty-turn workload still takes 3.31 s at p50 against the reference server's 1.63 s; what would close it, measured before building.
- **Setup:** Qwen3.8-27B Q8_0 on one MI50 at default clocks, cores 12-15 of the MI50 machine, the workload of the message-boundaries block, `--max-seqs 8 --ctx-size 32768`, each tree built from its own sha in the same image; main `c4305c0a` repeated run 7 (regenerate 2.96 s, edit 3.31 s at p50).
- **What the 3.31 s is made of**, at main with the default 10 GiB host tier:
  - three users of six lack the boundary before message 2, which the thinning to four drops on equal spacing while it keeps the first, message 1's: they fork 512 tokens in and read 800 to 940 tokens in 3.6 to 4.3 s, against 441 to 600 tokens in 1.9 to 3.1 s for the users who have it;
  - a request waits about 350 ms behind one job pass in flight, 64 rows at the prompt rate, on about a third of the requests (the server's own queued time);
  - boundaries on whole blocks cost the users with the right boundary 25 to 37 tokens beyond the reference's read, about 0.1 to 0.15 s, so a boundary at the message's exact start, which would need the partial block's KV copied with the state, is not worth building for this;
  - the prompt rate: about 0.3 s slower than the reference per 430 tokens.
- **More boundaries a conversation**, up to 64 instead of four (measurement builds, not merged), edit p50 at a 10 GiB and a 16 GiB host tier:

  | which boundary the room takes | 10 GiB | 16 GiB |
  |---|---:|---:|
  | the oldest | 6.1 s, no boundary forked | 6.1 s, no boundary forked |
  | a conversation's most crowded, the longest unheard first | 3.77 s, one conversation left with none | 2.81 s |
  | the cheapest of the conversation holding most | 6.14 s, no boundary forked | 2.54 s, four users of six; regenerate 2.07 s |

  At the default size, half of what the host has free here, the host tier holds about four 150 MiB states a conversation beside the copies, so keeping more helps only with more room; whether the host tier's default or a smaller state should give it that room is open.
- **An idle job's pass capped at its chunk** (`fix/job-idle-chunk`, test and change, not merged): interleaved and repeated against main `6a365411`, edit p50 3.36 and 3.31 s against 3.32 and 3.32 s, follow-ups p50/p99 2.40/4.07 and 2.38/4.08 s against 2.36/3.71 and 2.36/3.71 s, 44 requests of 114 queued about 354 ms against 38 at 349 ms, the job's completion after its reply 0.10 s at p50 on both. Most of a job's rows are read while its reply is written, already at 64 rows a pass, and few at idle, so the cap barely changes what a request meets, and split into more passes the idle remainder overlaps more arrivals; a chunk of 16 rows is worse on every count (edit 3.57 s, follow-ups 2.56/4.42 s, waits of 470 ms). The 350 ms wait is one 64-row pass, not an idle job's whole budget.
- **Left:** the host room for boundaries (the host tier's default or a smaller state), and the prompt rate; nothing of this lands as code.

## Speculative decoding in the server (2026-10-02, branch feat/spec-server, step 5 of SPECULATIVE, lands by fast-forward)

- **Design, against section 3 as the code now stands** (proposed in the devlog before building):
  - a decoding request with a mark and drafts is one verify entry [last pick, d1 ... dk], extent 1, every row's logits; its rows are sampled in order by the request's own sampler as `infer::accept` does, and the history retracted right after, before park, pause or fork;
  - the embedded drafter's chains run as one batched draft on the head's device for every request drafting that round (`Model::draft` over several sequences), after a pass is sampled and before the next is formed; lookup drafts on the host;
  - `spec::draft_length` also takes the decode columns a pass has left (the device profile's widest decode build), so drafting fills idle columns and is off under load, and a verify stays inside the room its request's reservation holds; at most 64 draft rows a pass;
  - logits rows 2 x (max_seqs + 64), rows a pass reserves ubatch + max_seqs + 64, `Request::kRowsWaiting` at least draft_max + 1, a mark a request at once, `mark_rows` draft_max + 1;
  - the host tier carries an embedded drafter's carried row with its checkpoint slot;
  - jobs, replays and resumes never draft; `--drafter` and `--draft-max` on `serve`, no request field.
- **Changed after the first figures** (proposed in the devlog, 2026-10-03 01:02): the decode columns were a stand-in for section 3's pass cost model and were wrong on the 27B, whose pass costs about 10 ms a row past a few rows; a pass now drafts what a measured price pays for (`spec::PassTimes`, `spec::draft_depths`), the columns only bounding it. Eight marks counted before the budget left the 27B on one MI50 1536 of 8192 KV tokens and no checkpoint, so eight users queued behind four; marks past the first now fit only the room the budget and the checkpoints leave (`PlacementRequest::fit_marks`), and no more requests draft at once than marks are free.
- **Done:**
  - the design above, built: verify entries in `Scheduler::form`, drafts asked once a pass (`Scheduler::propose`, `spec::Proposer::draft_all`), the embedded drafter's chains in one batch (`Model::draft` over `DraftAsk`s, longest chain first, the carried rows gathered by slot), `Backend::decode_columns` (the Vulkan profile's narrower decode build, 1 on the CPU), `server::draft_marks`, sampling a verify's rows with `infer::accept`, the retract right after `end_pass` (`settle_verifies`), the host tier carrying the drafter's carried row, `/v1/health`'s `drafted` and `kept`, `serve --drafter` and `--draft-max`, `server_mix_check.py --drafter`;
  - tests: `server-spec` (lookup on the synthetic Q8_0 model over one and two CPUs and paused, the hybrid MTP model's embedded drafter and lookup over one and two, greedy, sampled, capped, uncapped and stopped, cancelled in flight, each reply its reply alone without drafts), `arch-qwen35` (the host tier's carried row; three histories drafted in one batch, each its drafts and rows alone), the `qwen35` component's served drafts;
  - at 8460c60b, before the batch, on an MI50 (GPU[7], cores 4 to 7): CTest 43 of 43, the device suite's qwen35, server and decode-probe, and `server_mix_check.py` on Qwen3.6-27B-MTP Q8_0 with the embedded drafter on one MI50 and over two at two passes in flight, every request equal alone, together and skewed and to the CLI without drafts.
  - at 3f9e35f5 (the batch) on the Radeon VII: CTest 44 of 44 and the device suite's qwen35, server and decode-probe (XDEV's 30B captures overlapped the CTests; nothing was timed).
  - at dd90e175 (the price) on an MI50: CTest 43 of 43, no compiler warning, the device suite's three components, and `server_mix_check.py` on Qwen3.6-27B-MTP Q8_0, every request equal alone, together, skewed and to the CLI, with the drafts fed and kept now printed and required: embedded on one MI50 1033 fed, 670 kept, over two at two passes 1082 and 715, lookup on one 392 and 149; the device tier against main b7a6d235 (devtier.sh): the CPU and device suites 23 components PASS each, raw-blocks failing on the container's missing numpy and baseline skipping without its models, and the Qwen3-0.6B and Qwen3.5-0.8B identity cells 14 of 14 the same on the CPU and on the device.
  - **The final serving matrix**, at 0651303a (the code of 47336670), interleaved per placement in the order llmx embedded, reference without drafts, llmx drafts off, reference draft-mtp, then the reverse: Qwen3.6-27B-MTP Q8_0, `server_load.py`, the eight fixed prompts, 256 tokens a reply with the end ignored, greedy, `--max-seqs 8 --ctx-size 8192` (reference `-np 8 -c 8192`, `-sm layer -ts 1,1` on the split, whose verbose start of the same commands logs pipeline parallelism enabled, with and without draft-mtp), drafts at 3 (reference `--spec-type draft-mtp --spec-draft-n-max 3`), cores 4 to 7, each arm started below 55 C at default clocks, three rounds an arm, every round kept, output tok/s at 1 / 2 / 4 / 8 / 16 users (16 queue behind 8):

    | one MI50, in the order run, every round | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | llmx embedded, run 1 | 40.8 / 40.8 / 40.0 | 41.9 / 43.9 / 42.6 | 54.5 / 55.9 / 56.1 | 74.7 / 75.0 / 74.2 | 74.6 / 74.8 / 75.1 |
    | reference without drafts, run 1 | 21.4 / 21.8 / 21.8 | 33.9 / 31.2 / 30.4 | 47.5 / 44.3 / 45.0 | 52.6 / 55.6 / 55.7 | 56.3 / 54.8 / 54.6 |
    | llmx drafts off, run 1 | 22.8 / 22.6 / 20.5 | 33.5 / 33.0 / 32.5 | 53.3 / 53.8 / 53.4 | 67.2 / 68.6 / 66.7 | 66.2 / 66.2 / 66.1 |
    | reference draft-mtp, run 1 | 34.1 / 34.7 / 34.1 | 31.3 / 30.7 / 29.7 | 27.2 / 27.0 / 27.1 | 30.2 / 30.5 / 30.5 | 30.6 / 28.6 / 29.5 |
    | reference draft-mtp, run 2 | 33.1 / 34.4 / 34.3 | 31.3 / 30.9 / 30.0 | 26.8 / 26.9 / 26.9 | 30.8 / 30.1 / 30.2 | 30.9 / 30.6 / 29.4 |
    | llmx drafts off, run 2 | 20.6 / 20.4 / 19.6 | 31.2 / 30.4 / 30.3 | 51.2 / 51.2 / 50.7 | 63.1 / 63.4 / 63.9 | 64.1 / 66.4 / 65.7 |
    | reference without drafts, run 2 | 21.2 / 21.8 / 21.8 | 33.6 / 30.6 / 30.1 | 46.7 / 45.3 / 45.0 | 59.1 / 57.1 / 57.3 | 58.7 / 58.0 / 57.7 |
    | llmx embedded, run 2 | 40.8 / 40.7 / 37.5 | 38.6 / 38.4 / 36.8 | 46.2 / 46.7 / 45.8 | 62.7 / 62.2 / 62.3 | 62.4 / 62.4 / 62.7 |

    | one MI50, mean of the two runs' best rounds | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | reference without drafts | 21.8 | 33.8 | 47.1 | 57.4 | 57.5 |
    | reference draft-mtp | 34.6 | 31.3 | 27.1 | 30.7 | 30.7 |
    | llmx drafts off | 21.7 | 32.4 | 52.5 | 66.3 | 66.3 |
    | llmx embedded | 40.8 | 41.3 | 51.4 | 68.9 | 68.9 |
    | llmx's best against the reference's best | +18% | +22% | +12% | +20% | +20% |
    | llmx embedded against llmx drafts off | +88% | +28% | -2% | +4% | +4% |

    | two MI50s, layer split, in the order run, every round | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | llmx embedded, run 1 | 41.7 / 41.7 / 41.4 | 41.1 / 42.7 / 43.8 | 71.4 / 71.2 / 71.9 | 116.7 / 114.3 / 115.0 | 116.1 / 116.3 / 116.0 |
    | reference without drafts, run 1 | 20.9 / 21.4 / 21.4 | 33.6 / 32.5 / 33.0 | 50.4 / 48.3 / 49.0 | 59.7 / 58.2 / 59.0 | 59.2 / 58.8 / 59.2 |
    | llmx drafts off, run 1 | 22.7 / 22.7 / 22.8 | 43.3 / 43.9 / 43.9 | 73.8 / 72.4 / 72.9 | 119.6 / 119.5 / 119.5 | 120.0 / 118.5 / 119.8 |
    | reference draft-mtp, run 1 | 33.2 / 34.0 / 34.2 | 31.5 / 31.2 / 30.5 | 27.8 / 27.3 / 25.4 | 31.1 / 31.0 / 31.6 | 30.5 / 30.4 / 30.6 |
    | reference draft-mtp, run 2 | 33.4 / 34.3 / 34.3 | 31.2 / 31.3 / 30.7 | 28.3 / 28.0 / 28.0 | 29.4 / 27.5 / 29.4 | 30.1 / 30.0 / 28.9 |
    | llmx drafts off, run 2 | 22.8 / 22.7 / 22.7 | 43.3 / 43.9 / 44.0 | 72.6 / 73.4 / 73.4 | 119.3 / 119.2 / 119.2 | 120.1 / 119.8 / 119.8 |
    | reference without drafts, run 2 | 21.1 / 21.4 / 21.3 | 33.2 / 33.0 / 31.8 | 48.0 / 45.9 / 47.6 | 51.1 / 47.6 / 48.5 | 53.0 / 53.1 / 52.7 |
    | llmx embedded, run 2 | 41.8 / 41.7 / 41.0 | 41.3 / 43.0 / 42.4 | 71.2 / 70.9 / 72.3 | 116.7 / 114.2 / 114.2 | 116.2 / 115.8 / 116.0 |

    | two MI50s, layer split, mean of the two runs' best rounds | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | reference without drafts | 21.4 | 33.4 | 49.2 | 55.4 | 56.1 |
    | reference draft-mtp | 34.2 | 31.4 | 28.1 | 30.5 | 30.3 |
    | llmx drafts off | 22.8 | 44.0 | 73.6 | 119.4 | 120.0 |
    | llmx embedded | 41.8 | 43.4 | 72.1 | 116.7 | 116.2 |
    | llmx's best against the reference's best | +22% | +32% | +49% | +115% | +114% |
    | llmx embedded against llmx drafts off | +83% | -1% | -2% | -2% | -3% |

    On one MI50 the second run of each llmx arm came out slower than the first (drafts off by 3 to 9 percent, embedded by up to 17 percent at 4 to 16 users) with the same drafts fed and kept, so the card, not the code, moved within the session; the means average it. On the split, where the runs agree to a percent, drafting gives up 1 to 3 percent from 2 users up, where the price finds the drafts barely pay, and llmx's best there is drafts off. The machine's one-minute load average at each arm's end was 5 to 17; no finer activity monitor ran.
  - Serving figures (earlier sessions), Qwen3.6-27B-MTP Q8_0, `server_load.py`, the eight fixed prompts, 256 tokens a reply with the end ignored, greedy, `--max-seqs 8 --ctx-size 8192` (reference `-np 8 -c 8192`, `-sm layer -ts 1,1` on the split, where a verbose start of the same command shows pipeline parallelism enabled), drafts at 3 (reference `--spec-type draft-mtp --spec-draft-n-max 3`), cores 4 to 7, each arm started below 55 C at default clocks, the best of three rounds, output tok/s at 1 / 2 / 4 / 8 / 16 users (16 queue behind 8). The rows at 7da75632 are the candidate's; the others show how it got there.

    | one MI50 | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | llmx drafts off, 7da75632 | 22.9 | 33.4 | 57.8 | 71.7 | 71.9 |
    | llmx embedded, 7da75632 | 41.2 | 43.2 | 54.9 | 71.6 | 71.6 |
    | change | +80% | +29% | -5% | 0% | 0% |
    | reference without drafts | 21.8 | 34.4 | 51.3 | 60.7 | 59.5 |
    | reference draft-mtp | 35.1 | 32.1 | 28.9 | 31.9 | 31.0 |
    | llmx off / embedded, columns only, 28de9130 | 22.8 / 41.1 | 37.9 / 43.1 | 59.1 / 52.3 | 70.2 / 56.8 | 70.1 / 57.1 |
    | llmx off / embedded, priced, marks before the budget, dd90e175 | 23.0 / 40.2 | 37.6 / 50.5 | 60.9 / 59.8 | 75.9 / 57.0 | 75.8 / 59.1 |
    | llmx off / embedded, fitted marks, 56c0dc5a, two pairs | 22.8 / 40.7 | 33.8 / 41.1 | 54.5 / 52.9 | 66.2 / 67.2 | 66.5 / 68.1 |

    | two MI50s, layer split | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | llmx drafts off, 7da75632, two runs | 22.9 | 44.0 | 73.3 | 119.2 | 120.0 |
    | llmx embedded, 7da75632, two runs | 42.1 | 43.2 | 72.4 | 117.1 | 116.3 |
    | change | +84% | -2% | -1% | -2% | -3% |
    | reference without drafts | 21.4 | 33.3 | 50.6 | 59.9 | 54.9 |
    | reference draft-mtp | 34.6 | 32.0 | 28.4 | 30.3 | 30.5 |
    | llmx off / embedded, priced by latency, dd90e175 | 22.3 / 42.0 | 43.9 / 38.0 | 73.4 / 56.8 | 119.8 / 116.9 | 120.0 / 115.9 |
    | llmx off / embedded, priced by retirements, a chain once a pass in flight, 72ee44d0, two pairs | 22.8 / 41.7 | 43.4 / 37.7 | 73.4 / 71.4 | 119.4 / 117.6 | 119.4 / 115.6 |

    Every round of the arms above, output tok/s, in the order run (the tables take each level's best):

    | arm, every round | 1 | 2 | 4 | 8 | 16 |
    |---|---|---|---|---|---|
    | one MI50, llmx drafts off, 7da75632 | 22.9 / 22.6 / 20.3 | 33.4 / 33.0 / 32.9 | 54.8 / 56.7 / 57.8 | 70.6 / 70.7 / 71.7 | 71.8 / 71.6 / 71.9 |
    | one MI50, llmx embedded, 7da75632 | 40.1 / 41.2 / 40.8 | 42.2 / 43.2 / 42.9 | 54.9 / 54.4 / 54.0 | 71.6 / 71.5 / 70.8 | 70.8 / 71.6 / 69.6 |
    | one MI50, reference without drafts | 21.5 / 21.8 / 21.8 | 34.4 / 33.4 / 34.3 | 51.3 / 49.2 / 48.5 | 60.7 / 58.9 / 58.4 | 59.5 / 55.3 / 58.1 |
    | one MI50, reference draft-mtp | 34.1 / 35.1 / 35.0 | 32.1 / 31.2 / 31.5 | 28.7 / 28.0 / 28.9 | 31.8 / 31.5 / 31.9 | 30.5 / 31.0 / 30.9 |
    | split, llmx drafts off, 7da75632, run 1 | 22.9 / 22.8 / 22.9 | 43.3 / 44.0 / 43.9 | 72.4 / 73.3 / 73.3 | 119.0 / 118.9 / 117.8 | 119.5 / 119.9 / 120.0 |
    | split, llmx drafts off, 7da75632, run 2 | 22.9 / 22.8 / 22.8 | 43.3 / 43.9 / 43.7 | 72.4 / 73.2 / 73.3 | 119.2 / 119.3 / 119.2 | 120.0 / 119.8 / 120.0 |
    | split, llmx embedded, 7da75632, run 1 | 42.1 / 42.1 / 41.5 | 41.5 / 43.0 / 43.1 | 72.6 / 70.6 / 72.0 | 117.4 / 114.8 / 115.2 | 116.0 / 116.3 / 116.1 |
    | split, llmx embedded, 7da75632, run 2 | 42.0 / 41.9 / 41.5 | 41.5 / 43.2 / 43.0 | 71.3 / 72.1 / 71.7 | 116.8 / 114.3 / 114.0 | 116.1 / 116.2 / 115.9 |
    | split, reference without drafts | 20.9 / 21.4 / 21.4 | 33.3 / 31.8 / 32.4 | 50.3 / 43.0 / 50.6 | 59.9 / 57.7 / 57.0 | 54.9 / 53.5 / 53.3 |
    | split, reference draft-mtp | 33.5 / 34.6 / 34.5 | 32.0 / 30.9 / 31.6 | 28.3 / 28.2 / 28.4 | 30.3 / 30.1 / 29.7 | 30.3 / 30.5 / 30.3 |

    The machine's one-minute load average was logged at the end of each arm (6 to 20, other users' work on other cores, once 93); no finer activity monitor ran, so activity during an arm is not known.
    Llmx with drafts against the reference with draft-mtp, the same depth: one MI50 +17%, +35%, +90%, +124%, +131%; the split +22%, +35%, +155%, +286%, +281%. Without drafts against without: one MI50 +5%, -3%, +13%, +18%, +21%; the split +7%, +32%, +45%, +99%, +119%.
    At 2 users on the split both requests drafted in every pass, so every pass measured had the same rows and the price stayed unknown (72ee44d0: -13%); 7da75632 measures a pass without drafts then. At 8 users on one MI50 the drafter's eight marks counted before the budget had left 1536 of 8192 KV tokens and queued four requests (dd90e175, time to first token 9.7 s against 0.7 s); 56c0dc5a fits them after it. The one-MI50 drafts-off arm at dd90e175 ran while the machine's load average was 93 (another user's work on other cores), and the one-MI50 figures drift by some 10 percent between sessions hours apart, so only pairs of one session are compared.
  - The drafter loaded and drafting nothing (`bench --model`, one MI50, 8 sequences, off, on, on, off at 56c0dc5a): 79.2, 76.7, 74.8, 72.6 tok/s, mirrored -0.3 percent, a falling card rather than a cost.
  - At 7da75632, rebased onto main b18acd9d, on an MI50: CTest 43 of 43, no compiler warning, the device suite's three components, and `server_mix_check.py` every request equal alone, together, skewed and to the CLI, with drafts fed and kept.
  - Against main b18acd9d at 7da75632: the served ids of Qwen3-0.6B Q8_0 through `server_mix_check.py --ids`, drafts off, the same on the CPU and on an MI50; a timing round without drafts, Qwen3.6-27B-MTP Q8_0 on one MI50, `server_load.py`, main, branch, branch, main, main, branch, each cool-gated at default clocks: 1 user 23.0, 22.7, 22.8, 23.0, 22.8, 22.8 tok/s (branch -0.7 percent on the means), 8 users 74.2, 71.7, 73.2, 73.2, 72.6, 72.9 (-1.0 percent), inside main's own spread of 2.2 percent at 8 users.
- **Landing:** XDEV's review closed at 47336670 (18:49) and the coordinator's found nothing more (19:23); a hosted run was green on the squash of 47336670 (37131480729). Main then took message-boundary checkpoints (c4305c0a), whose state alone in host memory and fork with a state now carry the drafter's carried row with the slot, held by an `arch-qwen35` case that fails without the copy; rebased onto it, CTest 39 of 39 and the CPU suite's docs, dead-code, cli, qwen35 and server pass, and the device checks and the hosted run run again at the landing head, which lands by fast-forward.

## Message boundaries: an edited or regenerated earlier turn read from its message (2026-10-03, branch feat/message-checkpoints, step 2b part b of SPECULATIVE, lands by fast-forward)

- **Done:**
  - **A state alone, and a fork with it** (`model/history.hpp`): `Model::save_host` without blocks copies a checkpoint's state alone (`HostHistory::blocks` false, `host_bytes(length, false)`), and `Model::fork(src, length, state)` shares `src`'s blocks below `length`, whatever checkpoint `src` holds, with the state copied back into a checkpoint slot of its own, which the fork's first pass reads in place; `restore_host` refuses a state alone.
  - **Message boundaries** (`server/scheduler.hpp`): each re-prefill job's checkpoint already sits where its conversation's next user message starts, whole blocks inside the prefix `chat::stable_prefix` checked, so as the job completes its state is copied alone to host memory (`keep_boundary`, a `Boundary` with its tokens and row classes), no staging slots; a conversation keeps `kBoundaries` (4), its first and newest among them, the one whose neighbours lie closest going; boundaries live in the host tier's room after superseded copies and before any other copy, and a conversation's boundaries take its age as it adds one, so the room the tier needs takes those of the conversation that went longest unheard.
  - `enter` takes a boundary that shares more than any donor or host donor (`best_bound`), with the blocks of a donor holding its rows, its tokens and row classes below its position (`holds`), or of a host donor holding them, promoted first, and a checkpoint slot, the boundary pinned while room is made; `admit` forks with its state, and a fork whose copy fails reads the history from the start.
  - A boundary's and a host donor's entry is made before its copy is enqueued, so no allocation falls between a copy and the entry that releases it through `Model::release_host` (the other developer's review; the host donors' case came with part a).
  - `/v1/health` gives `boundaries` and `boundary_hits`.
- **Tests:** `arch-qwen35`'s `host_state_fork`; `server-resume`'s `message_boundaries` on one CPU and over two, and `boundary_faults`, copies to host memory failing (no boundary or copy kept) and copies from it failing (the edit's fork with a state fails and reads from the start), each reply its reply alone and the ledger whole.
- **Measured:** Qwen3.8-27B Q8_0 on one MI50 (renderD135) at default clocks, cores 12-15 of the MI50 machine, the six-user workload of the host-tier-room block (six users of twenty turns taken in turn, then for each user a regenerate and an edit at turn 2, the third user message), `--max-seqs 8 --ctx-size 32768 --host-cache-bytes 10737418240` on both llmx arms, each built from its own tree in the same image, main `llmx 0.1.0+gb18acd9d9c29` and the branch `llmx 0.1.0+gc980cbd13cee`; run 7 ran the branch, main, then the reference server at its defaults; load average 7 to 10 on average, 15 at most:

  | arm | regenerate at turn 2, p50 s | tokens read | edit at turn 2, p50 s | tokens read | follow-ups at turns 10-19, mean s |
  |---|---:|---|---:|---|---:|
  | main, run 7 | 5.69 | 1299 to 1449 | 6.04 | 1313 to 1496 | 2.55 |
  | branch, run 7 | 3.00 | 403 to 937 | 3.36 | 441 to 939 | 2.56 |
  | reference server, run 7 | 5.18 | 1309 to 1449 | 1.63 | 341 to 563 | 3.06 |

  Every regenerate and edit on the branch forked a boundary (`boundary_hits` 16 with four follow-ups that forked one, 20 boundaries held at the end); for three users the boundary kept was the one before the third message, which they read alone, 403 to 600 tokens in 1.95 to 3.12 s, and for three the thinning had kept the one before the second message, 512 tokens in, so they read 801 to 939 tokens in 3.6 to 4.35 s.
  The reference keeps a state at any token and reads the edited message alone, 341 to 563 tokens, but reads every regenerated prompt whole; llmx's boundaries sit on whole blocks, where the job's ids end, up to 127 tokens before the message.
  The follow-up turns are unchanged (2.56 s against 2.55 s on average), and every llmx reply is main's, byte for byte; main's regenerate and edit in run 5, 5.70 and 6.05 s, agree with run 7's.
  A first build that let the copies drop boundaries before superseded copies kept none to the end (run 5), and one whose boundaries aged by when each was made lost the first boundaries of the conversations that were still going (run 6, three users of six); all three runs' files are on the test machine in the measurer's scratch directory, `mb/hr5`, `hr6` and `hr7`.
- **Gates** (server tier, on the MI50 machine's CPU in containers of 4 CPUs from the build image, each tree built from its own sha, the code `llmx 0.1.0+gca3a232f974b` against main `llmx 0.1.0+g32a17d89802f`): CTest 37 of 37, every component of the CPU suite, the qwen35 real-model checks skipping without their files, and Qwen3-0.6B Q8_0's greedy ids and logits main's; the other developer reviewed the code, found one allocation-failure ownership window, rechecked its fix (above) and found none left; the hosted run of the landing head.
- **Left:** the coordinator's review; the change lands by fast-forward once the hosted run is green at the head and main has not moved.
- **Gotchas:** the boundary of the first user message is not kept, since no job precedes it; an edit of turn 1 reads from the system prompt. A boundary is only usable while a copy of a later history of its conversation is on the devices or in host memory.

## Step 0 of the tensor split: a group's sum measured on 2 to 4 MI50s (2026-10-03, branch tools/tp-exchange, lands by fast-forward)

- **Done:** `llmx-vk-handoff exchange A,B[,C,D] [epochs] [device|host]`: every member writes its F32 partial into every member's inbox and adds the slots in member order, every sum checked against the exact sum; the dispatch floor with no peer; the exchange through sync files with one submission an epoch a member; each member's arrival on the host's clock through calibrated timestamps; and a flag wait inside one submission under the Vulkan memory model at device and queue-family scope, each spin bounded at 2^16 reads.
  Measured on the Linux machine's MI50s of one root complex (83:00, 86:00, 89:00), RADV Mesa 25.0.7, Linux 6.17.13, default clocks, 200 epochs a chain, median of 5 chains, no wrong sum:

  | width | inboxes | 20 KB | 160 KB | 1.25 MB | 10 MB | arrival spread at 20 KB |
  |---:|---|---:|---:|---:|---:|---:|
  | 2 | uncached device memory, dma-buf | 153 to 154 us | 164 to 166 us | 380 us | 2.2 ms | 60 us |
  | 2 | host memory imported into both | 151 us | 115 us | 489 us | 3.3 ms | 56 us |
  | 3 | uncached device memory, dma-buf | 224 us | 293 us | 933 us | 6.2 ms | 134 us |
  | 3 | host memory imported into all | 170 us | 239 us | 921 us | 6.7 ms | 93 us |
  | 4 | uncached device memory, dma-buf | 268 us | 433 us | 1.82 ms | 11.9 ms | 200 us |
  | 4 | host memory imported into all | 278 us | 457 us | 1.88 ms | 13.0 ms | 207 us |

  The four cards are 83:00, 86:00, 89:00 and 8c:00; the floor with no peer is 13 to 19 us at 20 to 160 KB; the pair 86:00 and 89:00 measures as 83:00 and 86:00.
  The flag wait never saw a peer's flag inside a submission at either scope, on either placement, at any size and width, nor with the spin polling by an atomic read-modify-write; the release stores and acquire loads compile to `glc` accesses with no L2 writeback or invalidate.
  Review found the first flag modules invalid SPIR-V: the device-scope variants, and the queue-family ones through their counters' implicit device-scope atomics, lacked the `VulkanMemoryModelDeviceScope` capability, and the unoptimized build skipped the validation that would have refused them, so the width 3 and 4 flag results are observations of invalid modules.
  The modules are now built optimized, so glslc validates each, with `#pragma use_vulkan_memory_model`, which makes glslang declare the capability, and with the counters' atomics at the variant's scope; with them, at width 2 on both placements and every size, every flag wait still timed out, and the sync-file figures matched (20 KB 126 us, 160 KB 146, 1.25 MB 377, 10 MB 2.2 ms), the calibrations' uncertainty at most 11 us beside spreads of 55 to 1100 us.
  The importing side of each inbox now takes the device-uncached type too, where the first runs took the first type an import allowed, a cached one; with both sides uncached, in device memory and in system memory shared by dma-buf, the flag wait still timed out at width 2, and host memory imported into a card offers only a cached type, so no Vulkan placement here gives an uncached view on both sides that a peer's writes reach inside a submission.
  `llmx-multi-device-bench stages` on one MI50: Qwen3-8B Q8_0 takes 14.6 ms of device time a decode pass over 364 dispatches, and Qwen3.6-27B Q8_0 44.8 ms over 740.
  Pass costs with `llmx bench` (ms a pass) for the serving model of `docs/TENSOR-SPLIT.md`, section 5: Qwen3.6-27B Q8_0 on one MI50 at 1, 8, 16, 32 and 64 rows 43.2, 107.2, 206.7, 438.2 and 1543, pp512 1972 a chunk, pp2048 8123; Qwen3-32B Q8_0 over two MI50s at one pass in flight 51.5, 141.4, 274.9, 496.8 and 1057, pp512 2503, pp2048 6834 pipelined.
  Outcome, agreed with the other developer under the user's delegation: the Vulkan collective and steps 1 and 2 are deferred, the exchange mode lands as the probe decision 2 requires on every new driver, and the plan's section 8 lists what reopens the work.
- **Left:** width 2 across root complexes was skipped by the coordinator, and the host's recording time a member a layer, which the stage timing does not separate, matters only once the work reopens.
- **Gotchas:** host memory imported into a card must be a whole number of the import alignment (4096 bytes here), so the tool rounds its inboxes to 64 KiB; glslang declares the device-scope capability of the Vulkan memory model only under `#pragma use_vulkan_memory_model`, and only an optimized build validates a module; a spin on another card's flag that is not bounded ends in a ring timeout; the tool refuses an epoch count or device list it cannot read before any allocation and fails a run with a wrong sum.

## Tensor split and the staged tensor split (planned 2026-10-03, branch docs/tensor-split-plan, design only)

- **Goal:** the plan for MULTI-DEVICE's phases 6 and 7 in `docs/TENSOR-SPLIT.md`: research of Megatron-LM, vLLM, SGLang and mx-llama.cpp, the op sharding for qwen3 and qwen35, two sums a layer in a fixed member order, the exactness rule, skew, serving and speculative decoding over a group, and the order of work, with the staged form designed in from the first step.
- **Done:** the plan, and one measurement on two MI50s on one root complex (PCI 83:00 and 86:00, RADV Mesa 25.0.7, Linux 6.17.13, default clocks), a standalone probe built from `tools/vulkan_handoff.cpp`'s helpers: a two-card exchange of F32 partials summed in member order costs 137 to 140 us at 20 KB through sync files over a 13 us dispatch floor, 137 to 167 us at 160 KB, 367 to 387 us at 1.25 MB and 2.0 ms at 10 MB, every sum correct, and a submitting thread per card does not lower it; a device-side wait inside one submission never saw the peer's writes (device-uncached dma-buf memory, host memory imported into both cards; shader stores, a host-release barrier, a buffer marker), so the writes are visible only at a submission boundary; the same wait with the Vulkan memory model (release and acquire atomics with MakeAvailable and MakeVisible at device and queue-family scope, the memory model features enabled) timed out too, and the ISA shows `glc` stores and loads with no L2 writeback or invalidate, which gfx906 lacks. The first run's unbounded spin cost each of the two cards one ring timeout with a soft recovery.
- **Left:** step 0 measured and step 0b moved `runtime.hpp`'s pass code into `model/passes.hpp`; step 0's outcome (2026-10-04, the plan's section 8) deferred steps 1 to 8.
  Reopened by the user on 2026-10-04: the tensor split is built on Vulkan now, not on a ROCm backend, and gated on the reference's ROCm tensor split (the plan's section 8, Built on Vulkan now), which section 2.8 records with its all-reduce, its environment and its measurements at widths 2 and 3 on Qwen3.6-27B and Qwen3-32B Q8_0 beside llmx on the same cards.
  Step 1 (`feat/tp-shard`) landed; step 2 (`feat/tp-cpu`, the CPU collective and `--tensor-width`) is in review; then step 3 (`feat/tp-vulkan`, the sync-file collective), each on its own branch with its gates; beside them the fast Vulkan transport research of section 8, item 3, the kernel-side route first, since every in-submission wait tried, the reference's own KFD-allocated uncached fine-grained VRAM included, fails on amdgpu's implicit sync of imported dma-buf (section 4.3).
  Width 4 of the reference waits for the fourth card; width 8 stays refused until a width-8 Vulkan sum is measured, and the 8-card gate is the reference's best shape there.
- **Checks:** the 2026-10-04 revision (Vulkan now, the ROCm gate, the collective, the width-8 rule and the kernel-side transport reading) passes the suite's docs and dead-code components with no new findings against their lists, their 16 and 18 planted faults caught; it is docs only and lands by fast-forward on the coordinator's review, the other developer away.
- **Gotchas:** a device-side spin on another card's flag must be bounded far below the ring timeout (10 s here); MULTI-DEVICE's Tensor split section now points to the plan and no longer says the Vulkan sum crosses through host memory.

## Host-tier usage and raw-decoder coverage wording (2026-10-03, branch docs/host-tier-usage-20261003, lands by fast-forward)

- **Done:** USAGE links to the host-tier eviction policy described in [SERVER](SERVER.md), replacing its old claim that eviction was strictly by age. CI and ROADMAP now include MXFP4 in the raw-decoder coverage lists, as `tests/roundtrip.py` already does.
- **Checks:** all 88 tracked Markdown pages reconciled against the code and existing reviewed pages: 75 identical canonical Git pages retain their review, and the 13 differing pages were checked against current main, keeping the unmerged IQ4 feature out of this correction. The docs and dead-code components pass with no new findings against their existing lists; their 16 and 18 planted-fault checks pass. The documentation-only gate requires these two components, without repeating runtime tests or waiting for hosted CI.
- **Gotchas:** no runtime behavior, default, flag or numerical gate changes; the IQ4 and message-checkpoint feature branches remain separate.

## One host copy per conversation, host memory kept for conversations that come back, and its default size (2026-10-03, branch fix/host-tier-room, step 2b of SPECULATIVE, lands by fast-forward)

- **Cause, measured on main:** six users of twenty turns each taken in turn, Qwen3.8-27B Q8_0 on one MI50, `--max-seqs 8 --ctx-size 32768`: each conversation held two full copies, the request's donor and the re-prefill job's donor beside it (step 2c), and with the previous turn's job donor three, each about 650 MiB at 8k tokens; with every copy competing for the same host room the oldest went first, and under users taking turns the oldest is the one needed next, so from turn 10 on most follow-ups read their whole history.
- **Done:**
  - **One copy per conversation** (`Scheduler::supersede`): a job's donor, once it has read the whole of its ids, supersedes the conversation's other donors, the one it forked, its request's (`Request::parked_`) and every donor and host entry whose tokens its own begin with, such as the previous turn's job's; they go to the front of their tier, so they go first when room is needed, and `write_back` copies no superseded donor to host memory, so they stay, for a regenerated reply, only while room allows.
  - **Conversations that came back keep their place** (`Scheduler::write_back`): a donor whose conversation did not come back (`Donor::back`: its request, or the request whose reply a job read, forked nothing a tier kept; an entry promoted is marked come back) takes only free room and the room of superseded entries and of entries whose conversations did not come back either, oldest first, unless the tier has refused as many such donors in a row as it holds entries (`host_refused_`), so users taking turns over more conversations than the tier holds keep hitting the ones it holds, and conversations that stop coming back still leave.
  - **The donor count** (`park`) gives up a superseded donor first, or one the parking job's donor supersedes, which it marks so, before the oldest, so a job completing under `--max-seqs` donors never evicts an unrelated conversation's donor while its own conversation's older copies stay (the other developer's review).
  - **A renewed host copy takes its donor's standing** (`write_back`): a donor whose history a host entry already holds, a job donor of a regenerated reply that came out the same, say, renews the entry as not superseded and come back if either is, where it kept the superseded mark the job had put on the old copy and went first under host pressure as obsolete (the other developer's review).
  - **Default size** (`server::default_host_cache`, `host_cache_default` in `server/policy.hpp`): what `--max-seqs` histories take at the most one request may hold, the model context or the KV pool, whichever is smaller, within half of the host memory free once the model is loaded, and none where every storage is on the CPU, as approved; each copy still leaves the host the reserve the fit keeps (`detail::host_room`, `CpuBackend::host_reserve`), and serve's startup line prints the size taken.
  - Plain LRU and the rule suggested for comparison (keep the tier's content when admitting would evict an entry newer than the one written back) are the same here: the donor written back is always the newest, the conversation just served.
- **Tests** (`server-resume`, on the synthetic Q8_0 model; each fails on main's scheduler and passes with the change, checked by building each failing variant):
  - a request needing one donor's room evicts the superseded request donor of a conversation read again rather than an older unrelated one, and with a host tier copies nothing to host memory (fails on main: the unrelated conversation's repeat reused 0 tokens, against 256);
  - after two turns each read again, every donor evicted at once leaves two host entries, the unrelated conversation's and the second job's (fails with superseding by the job's ids alone: 3 entries);
  - one request at a time over host room for two copies: a conversation that came back stays while three newcomers take each other's room, and its repeat forks its 384 tokens from host memory (fails under plain LRU: 0); over room for one, the first newcomer is refused and the second, the tier having refused as many as it holds, evicts the conversation that came back (fails without the bound: the second newcomer's repeat reused 0, against 256);
  - with two donors at most, a job completing gives up its conversation's request donor and not an unrelated conversation's (fails on f14c548a, before the fix: the unrelated conversation's repeat reused 0 tokens, against 256);
  - a regenerated reply that comes out the same, read again, renews its conversation's host copy, which then survives two rounds of host pressure, so the follow-up forks its 384 tokens (fails without the renewal's refresh: 0);
  - `server-passes` holds `host_cache_default` below the cap, at it, past it, at a sum that would overflow and with no free memory known.
- **Measured** (Qwen3.8-27B Q8_0 on one MI50, renderD135, default clocks, cores 12-15 of the MI50 machine, load average 10 to 20 from other work in the llmx arms and 26 on average, 99 at most, during the reference's; six users of twenty turns of about 450 tokens without reasoning taken in turn, then a regenerate and an edit at turn 2 for each, `--max-seqs 8 --ctx-size 32768`, 3 checkpoint slots; `--host-cache-bytes 4294967296` on every llmx arm, since the default reads the host's free memory at start and gave 4234 and 5038 MiB on two starts here; each tree built from its own sha in the same image, main `llmx 0.1.0+gb7a6d2350785`, superseding alone `llmx 0.1.0+g4013b2b27540`, both rules `llmx 0.1.0+g4773c4ee3fd9`; the reference server at its defaults, 8 slots over a 32768-token cache with flash attention; every llmx arm gave the same reply bytes; time to first token, p50 per range of turns, and follow-ups that read their whole history):

  | arm | turns 1-9 p50 s | whole | turns 10-14 p50 s | whole | turns 15-19 p50 s | p90 s | whole | turns 10-19 mean s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | main, run 2 | 2.79 | 21 of 54 | 26.80 | 25 of 30 | 40.16 | 44.99 | 27 of 30 | 30.07 |
  | main, run 3 | 2.79 | 21 of 54 | 26.91 | 25 of 30 | 40.09 | 44.98 | 27 of 30 | 30.15 |
  | superseding alone, run 2 | 2.20 | 3 of 54 | 2.36 | 0 of 30 | 3.48 | 42.81 | 10 of 30 | 9.18 |
  | both rules, run 3 | 2.20 | 3 of 54 | 2.35 | 0 of 30 | 2.76 | 37.89 | 4 of 30 | 5.18 |
  | reference server, run 3 | 2.43 | 0 of 54 | 2.91 | 0 of 30 | 3.27 | 3.58 | 0 of 30 | 3.07 |
  | landing code at its default size (10082 MiB), run 4 | 2.20 | 3 of 54 | 2.35 | 0 of 30 | 2.69 | 3.11 | 0 of 30 | 2.54 |

  Run 2 ran superseding alone then main, run 3 both rules, the reference, then main, and run 4 the landing code, `llmx 0.1.0+g2c250a00ffd8`, with the review fixes below and without `--host-cache-bytes`, its default taking 10082 MiB, half of the host's free memory, load average 5 to 22; main's two runs agree within 0.1 s a range, and every llmx arm of the three runs gave main's reply bytes.
  At its default the landing code reads no follow-up's whole history from turn 10 on, 2.54 s on average over turns 10 to 19 against the reference's 3.07 s.
  Main's follow-ups that hit host memory read about 400 tokens, as the branch's do; the rest read 5k to 9.7k tokens at 22 to 47 s.
  The three whole re-reads at turn 1 on every llmx arm are first follow-ups that came while their job was still reading the reply (896 job rows during each), 850 tokens at 3.7 s.
  With both rules the one conversation of six that does not fit 4 GiB from turn 16 on, where six conversations of 8.3k tokens or more take 4.1 GiB and more, reads its whole history each turn and the other five keep hitting; with superseding alone three of six read theirs each turn from turn 17, each evicting the next one needed.
  The regenerate and the edit at turn 2 read their whole 1.3k to 1.5k tokens on every llmx arm, about 5.7 and 6.1 s at p50, since the conversation's copy holds its last checkpoint, past turn 2, which is part (b)'s case; the reference regenerates in 5.2 s reading its whole prompt and takes the edit in 1.6 s, reusing 964 tokens and reading 412.
  Job rows (`reprefill_rows`) during the timed follow-ups at turns 10 to 19: main 89728, superseding alone 27712, both rules 19968, a job reading a whole history again where its conversation missed.
- **Why the default changed:** a promoted copy stays in host memory, so host memory must hold every conversation the server resumes: here 6 x (150 MiB of state + 64 KiB a token) in whole 64 MiB slabs, about 4.5 GiB at turn 19, which a replay of the measured trace through the two rules leaves without a whole re-read from 5 GiB on. The old default, a quarter of the host's free memory after loading, gave 4.1 and 4.9 GiB on two starts of this host; the new one takes 8 x 1.5 GiB, the host bytes of the 22592 tokens one request may hold here, within half of the free memory, 8.3 to 9.8 GiB on those starts. The slabs stay with the server once allocated, so the default is what it may hold.
- **Gates** at `3b56f2f5` on main `b7a6d235` (server tier, the CPU of the MI50 machine): CTest 37 of 37, every component of the CPU suite, Qwen3-0.6B Q8_0's ids and logits main's, the test commit failing on main's code, and the hosted run; the other developer verified both review fixes and the default. Rebased onto main `087039ab` with a conflict only in this file, so the builds, CTest and the hosted run ran again at the head.
- **Left:** 2b part (b), message-boundary checkpoints for an edited or regenerated earlier turn (about 6 s on this workload against the reference's 1.3 to 3.1 s for an edit), follows.
- **Gotchas:** a promotion is sometimes made twice for one request, the second copying the same entry back after the first promoted donor was evicted again before the request was admitted (152 promotions for 132 requests with both rules); seen in the logs, not investigated here.

## macOS Intel CI timeout (2026-10-03, published)

Published at [59d7e14b](https://github.com/mxxm-t/llmx/commit/59d7e14b75364dd11cd520d21dfcc49ea801484e) after [CI 37115070878](https://github.com/mxxm-t/llmx/actions/runs/37115070878) passed all seven jobs. Both main refs were verified at that commit, and the owned temporary gate branch was removed. The following pre-publication record retains the original timeout and its validation; the separate half-weight numerical hold is unchanged.

- **Done:** the workflow changes only the macOS budget from 25 to 40 minutes. GitHub's check annotation confirms the timeout. Compared with the preceding passing main run, the build took 15m08s instead of 3m56s, and native checks took 6m28s instead of 3m21s. The added half-weight native test took only 5.32s; other native and Python checks also slowed. The dated evidence in [CI.md](CI.md#macos-intel-timeout-2026-10-03) records the observations and the bounded estimate.
- **Checks:** docs and dead-code passed with their unchanged known-finding and planted-fault counts. All 88 tracked Markdown pages were reconciled with the published-main review: 86 unchanged pages carried forward, and the affected CI/STATUS claims checked directly. The workflow diff is exactly one value; runtime, tests, build commands and individual test timeouts are unchanged. No local build or model test was run for this scheduling change.
- **Left:** exact-head hosted CI before fast-forward landing. The cancelled integration run remains retained; this checkpoint does not claim the longer job has completed.
- **Gotchas:** no failed or unfinished check becomes a pass. Preserve the cancelled run, keep every test and its own timeout, and leave Windows, Linux and HF job budgets unchanged. Hosted runner slowness is plausible, not proven by machine telemetry. The separate half-weight numerical hold is unaffected.

## Qwen3-MoE file-exact HF reference (2026-10-03, published)

Published at [d9c37b07](https://github.com/mxxm-t/llmx/commit/d9c37b072b27c0dfab6218233c978aab4aeb8fca) after [CI 37109015725](https://github.com/mxxm-t/llmx/actions/runs/37109015725) passed all seven jobs. Both main refs were verified at that commit. The tools release is complete; the following pre-publication record preserves its validation and the separately scoped runtime-qualification work.

The publication reconciliation reviewed all 88 tracked Markdown pages against the unchanged published tree and retained landing records; STATUS and CI needed publication updates, while the other 86 pages are unchanged. Docs and dead-code pass with their existing 13 and 7 known findings and all 16 and 18 planted faults. No runtime gate or failed numerical comparison is relabeled by this documentation correction.

- **Done:** shared expert/router mapping in `tools/gen_baseline.py`, using independent spec decoders without full-model copies. The existing layered owner runs Qwen3-MoE through HF's own forward with temporary load/release hooks. Mixed storage/order and version-refusal checks join the hosted reference-generator component. An isolated local environment with torch 2.5.1+cpu, transformers 4.55.2 and numpy 2.2.6 passes the tiny full-versus-layered check, global rotary/one-layer residency assertions and failure recovery. The final hand check compares 57,568 F32 values exactly, including checkpoint loading. At the original checkpoint, reference-generator passed 37/37; docs and dead-code pass with their 16 and 18 planted faults. Original goldens remain unchanged; the gated HF comparison retains its existing 2e-5 bound (largest difference 1.70921e-6), while the near-tie difference 1.36668e-6 remains diagnostic. No runtime policy or acceptance bound has changed.
- **Done:** the CPU tools-tier checkpoint `c7911709` passed 37/37 native tests and reference-generator, reference-consumer, docs and dead-code. The landing tests-first commit `f4f4c089` reproduced the layered CLI's late Windows `resource` import failure and Darwin's incorrect RSS units. The owner now reports the Windows lifetime peak working set and converts each POSIX platform's units correctly. The CLI also establishes offline mode before importing the Hub, explicitly requests cached tokenizer/config/model files, and gives the tokenizer writer its local file without changing the standalone generator's download behavior. Windows reference-generator passes 42/42; the tiny HF check still matches all 57,568 values exactly. Neither fix changes model arithmetic. At final source checkpoint `d6047e2b`, Linux reference-generator, reference-consumer, docs and dead-code all pass; the 42-test generator run skips only its Windows short-path case. The earlier native gate covers unchanged C++ sources. An initial scratch launcher quoting error ran no tests and remains recorded separately.
- **Left:** run exact-head hosted CI; source review found no blocking numerical defect. Complete the separate real-model qualification. The pinned Unsloth model card identifies Qwen/Qwen3-30B-A3B; config/tokenizer revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` matches the GGUF dimensions, all 151,669 HF token IDs, all 151,387 merges and the fixed 247 excerpt IDs. This is the fetched metadata revision, not a claimed original conversion revision. BOS/PAD defaults differ, so compatibility here covers only the declared unpadded inputs without added special tokens. At source `91a7a6ff`, one fixed 247-token real-file forward completed all 48 layers and wrote 37,528,192 finite F32 logits (SHA256 `2e1d01f1ba0320412b8223d11cc125104de0af629783bf12c34290725dade650`). It took 180.766 seconds including file hashing and reported a 3.116 GiB peak working set. The retained worker record pins the model, metadata and source; all six retained CPU/MI50 batched/decode captures pass the existing continuous mean-NLL delta bound of 0.01 (largest delta 0.003104030448). All six have 246/247 strict top-one diagnostic matches; adjusted top-five misses remain at positions 116 or 243 in four captures. This is not a completed release gate: six standard prompts, six distinct reset windows and disposition of the original supplemental CPU/device failure remain separate work. Original live memory samples followed the Windows launcher rather than its worker, so the peak comes from the actual worker's lifetime counter; both processes exited, and no live abort coverage is claimed.
- **Landing:** land this tools-only feature by fast-forward after exact-head hosted CI passes and both main tips remain unchanged. The final documentation reconciliation covers all 88 tracked Markdown pages; the reporting/offline fixes and reference math retain their measured source, and this final amendment changes documentation only. The real-model reference campaign and the retained 30B numerical criterion remain separately reported; neither is approved by the generator's tests.
- **Gotchas:** the pinned 30B UD file widens to 113.741 GiB of F32 weights, while its largest decoder layer is 2.321 GiB. The ordinary full-model loader cannot fit the rig. Preserve existing goldens and Qwen3.5 behavior; no precision allowance or golden has changed, and the real check used the existing file without downloading model weights. The authorized isolated CPU test dependencies are recorded separately; global Python is unchanged. The Qwen3-MoE hand check is not claimed as hosted coverage.

## Uncapped server check counts host-cache resumes (2026-10-03, published)

Published at [da38d13a](https://github.com/mxxm-t/llmx/commit/da38d13a82ed07fb77b97944bd0e501f3a97314f) after [CI 37104221441](https://github.com/mxxm-t/llmx/actions/runs/37104221441) passed all seven jobs. The following pre-publication record retains the original failure and validation; its hosted check and fast-forward landing are complete.

- **Done:** the retained Radeon integration failure reached equal replies and log-probabilities, then reported two pauses, nothing active or paused, three host promotions, and no device takebacks or recomputed rows. The scheduler promotes a host donor before admission and counts it in `host_hits`; this is a valid resume path omitted by the assertion. Its source is unchanged between main `b7a6d235` and integration `6c5e21aa`.
- **Change:** the check captures the full health record before the concurrent group and requires a new pause, no active/queued/paused requests afterward, and an increase in device takebacks, host promotions or recomputed rows. The existing reply and log-probability equality checks stay unchanged.
- **Checks:** the corrected real Qwen3-0.6B Q8_0 `check_uncapped` passed on the CPU and Radeon VII, with unchanged reply and log-probability comparisons. Both began with zero pause/resume counters and ended with no active, queued or paused requests: the CPU added two pauses and 649 recomputed rows; the Radeon added two pauses, 62 recomputed rows and three host promotions. These focused checks used the frozen integration executable `6c5e21aa`, not a new build of this test-only branch. Their full health records remain in `.tmp-half-final-20261003/server-resume-cpu/complete.json` and `server-resume-corrected/complete.json`. The docs and dead-code components passed; all 88 tracked Markdown pages were screened for affected claims, with the unchanged historical records retained.
- **Gate:** a fresh GCC 14.2 Release/Vulkan build at checkpoint `3ef65957` passed all 42 native tests, with no skipped tests. The complete server component then passed separately on the CPU and MI50, with no skipped subcases, including the real Qwen3-0.6B Q8_0 concurrent replies, log-probabilities, uncapped pauses/resumes and host-cache follow-ups. Source and binary hashes remained unchanged; both server children drained and the owned device was released. The final amendment updates this record only; the tested assertion and runtime are unchanged.
- **Landing:** this test-only branch lands as one commit by fast-forward after hosted CI passes at its final head and both main tips are verified unchanged. The original failing test already exists on main, so no duplicate test of its assertion is added. Hosted CI remains the final pending check; the original full integration suite is not relabeled as passed.
- **Gotchas:** this changes only test accounting, not runtime behavior or numerical bounds. The initial failure remains in `.tmp-half-final-20261003/suites/windows-results/suite.log`. Counters from the earlier requests run alone must not satisfy the concurrent group's witness. The separate half-weight source stays frozen; its supplemental 30B numerical hold is unchanged.

## A re-prefill job no longer takes whole passes beside a request in flight (2026-10-02, branch fix/job-idle-passes, lands by fast-forward)

- **Bug:** a job begun at idle (a finished reply read again for the next turn, step 2c) took a pass's whole budget whenever the pass being formed held no request rows. With passes in flight a decoding request is always in the other pass, so every pass the job formed counted as idle: on production (Qwen3.8-27B Q8_0 over two MI50s, two passes in flight, a 512-row budget) a long conversation's job read 512 rows a pass for minutes, over 23000 rows, and the user's reply ran at a token every 3 s, each token waiting for a 512-row prompt pass on both stages.
- **Fix:** a pass is idle for a job only while no request is active, in flight or not (`Scheduler`'s pass formation); a job begun while its reply was written still takes at most `kJobChunk` rows of a busy pass, and one begun after it waits.
- **Test first:** `server-resume` over a two-CPU split with two passes in flight and a 256-row budget: a job begun at idle beside a request decoding 200 tokens reads nothing from that request's first pass to its last and completes once it has ended, and a job begun while its reply is written reads beside that request at most 64 rows a pass and some; each request gives its reply alone. Both fail on main's code, where a pass carried 256 of the job's rows. The other developer's review asked for the two cases apart.
- **Production** ran with `--passes 1` until this landed, which keeps a decoding request in every pass formed.
- **Gates** (server tier, on the CPU of the MI50 machine, each tree built from its own sha): CTest 37 of 37, every component of the CPU suite, Qwen3-0.6B Q8_0's greedy ids and logits main's; the test commit fails on main's code (a pass carried 256 of the job's rows) and passes with the fix; the other developer reviewed the fix and the test. Rebased onto main `f22367c6` without a conflict, so the builds, CTest and the hosted run ran again at the head.

## Vulkan weight dispatch ownership (2026-10-02, published)

Published at [5835886c](https://github.com/mxxm-t/llmx/commit/5835886c92250317a69e7139ec7164f13874456d) after [CI 37034673513](https://github.com/mxxm-t/llmx/actions/runs/37034673513) passed all seven jobs. Both main refs and removal of its temporary gate were verified. The following pre-publication record preserves the integration measurements and their limitations.

- **Done:** one private Vulkan weight descriptor now selects support, row layout and modules, activation twins, float/BF16 and integer tiles, and existing dense/routed crossover families. One macro list generates the kernel IDs, diagnostic names and module bindings. An independent mechanical comparison against `fb366b16` preserves all 91 numeric IDs and their complete names/source/binding/count-array/preservation mappings. Shader sources and CMake entries are unchanged. The source dead-code check reads the list, with planted orphan-ID, missing-module and missing-name faults.
- **Development checks:** a fresh Visual Studio 18 2026 / MSVC 19.50.35728.0 Vulkan build with SDK 1.4.357.0, two workers, built `llmx`, `llmx-backend-vulkan-test` and `llmx-vulkan-quantization-test`; the executable reports `llmx 0.1.0+gfb366b16cef4.dirty`. On the Radeon VII, `backend-vulkan` passed in 30.28 seconds and `vulkan-quantization` in 13.68 seconds. The new test covered 331 independently expected dispatch/refusal cases; existing coverage passed 93,240 shared matrix precision values, 36 actual-path witnesses and 12,125 equivalent-extent pairs. These are correctness checks, not performance measurements. Optional float-preservation refusal subcases retain their documented skips. The source dead-code check passes with all 18 planted faults, and the docs check passes with all 16 planted faults. The affected owner pages and Markdown references to removed helpers were reviewed; the quantization plan now distinguishes existing safety owners from the deferred model-specific crossover optimization.
- **Completed correctness:** clean private `4ad79007` passes Linux native 42/42 and the strict device suite 25/25, including the actual RADV integer-dot path; Windows native 43/43 and docs/dead-code/architecture checks pass on the Radeon VII. Against base `fb366b16`, 22 CPU/device model cases give 66 byte-identical captures and matching records apart from build identity, covering all eight implemented weight types, dense and routed models. All 99 compiled shader modules match. Platform and optional hardware subcase skips remain explicit in the [checkpoint report](benchmarks/vulkan-weight-dispatch-20261002/report.md), which links the retained commands, hashes and logs. The clean Windows executable is `.tmp-vulkan-dispatch-windows-20261002/build/Release/llmx.exe`, version `0.1.0+g4ad79007fd2d`, SHA256 `fff71543ef89f77756f2176173baf4aeb8c7f1a6bf7ea4495830beb33e160fd2`; the preceding dirty build is development evidence only.
- **Timing assessment:** all 54 calls returned 0 and remain separated as the original 24, a rotated 24-call follow-up with identical binaries, and six dense-MI50 calls at ten internal repetitions. The original Qwen3-30B-A3B Q4_K_M CPU decode deficit (-49.28%/-51.60%) did not reproduce: follow-up medians are candidate 13.535, base 13.425 and control 13.550 tokens/s, with candidate/base pairs +0.52%/+1.12%. Dense MI50 decode at ten repetitions gives 392.875/390.855/392.410 tokens/s and candidate/base pairs -0.23%/+1.28%. Accept the bounded timing screen at measured repeatability, preserving unfavorable samples and mixed GPU prefill signs. This is not a speedup or proof of exact nonregression. The [report](benchmarks/vulkan-weight-dispatch-20261002/report.md) and [data](benchmarks/vulkan-weight-dispatch-20261002/results.json) retain every rate, the ten-repetition calls' standard deviations, activity/unknown observations and hashes. No fresh mx campaign or reference parity is claimed; that comparison remains a phase-level gate.
- **Diagnosis and limits:** candidate/control binaries have identical addresses, sizes and bytes for all 159 compared CPU/q8 functions, 22 bench functions and the 432-byte CPU vtable. Selected base/candidate Q4/Q5 decode instructions normalize identically, with constant/data and relocation limits documented. The added calls record child CPU/fault/I/O statistics and zero increments in cgroup memory max/OOM/OOM-kill counters. Their cold and cached reads do not establish the specific cause of the original deficit. One layout control does not establish a universal noise band. The final measurement container stopped at 18:50:46 local; the selected device reported 0% busy and 10,932,224 bytes of VRAM.
- **Integration:** squashed onto published main `2c293678`; only this status file conflicted, and both feature records are preserved. The inherited source/test/CMake patch is exact after removing hunk positions. Fresh detached builds at `d4db2515` pass Windows 43/43 and Linux 42/42 native tests, no native skips, all 331 dispatch/refusal cases on each, and docs/dead-code/architecture checks. All 99 Linux shader modules match the measured candidate. The original suites, identities and timing retain their recorded source scope under the conflict-free rebase rule. The final amendment changes documentation only; hosted CI checks that final commit. The Linux container is stopped with no remaining device clients; hashes and conditional hardware subcases are in the report data.
- **Landing:** lands as one commit by fast-forward when the integration checks and hosted CI pass and both main tips remain unchanged. The landing commit and its hosted run identify publication without a separate merge-record commit.
- **Gotchas:** no new storage types, shader expressions, CMake shader entries, numerical policies, thresholds or model-specific row classes. The independent mixed-expert capability fix is inherited from main; it was absent from the original measured candidate. Existing IDs, pipeline variants, refusal text and dtype witnesses must stay unchanged. Activity flags describe observed contention and coverage limits; they neither establish a cause nor waive the existing-path speed gate.

## Mixed routed projection types checked at load (2026-10-02, published)

Published at 530b10bf after all seven jobs in hosted run 37019138272 passed. The record below preserves the pre-publication validation.

- **Done:** test-first commit `8db68602` builds and fails both regressions: `model-validation` accepts the unsupported pair, and `arch-qwen35` omits its required operation. The repair declares `Op::mixed_experts` through one shared helper over the two declared projection roles; CPU supports it and Vulkan does not. The existing runtime operation check refuses an unsupported resident placement before allocation/adoption and leaves an unsupported stream destination's layer on its capable host.
- **Validation:** clean candidate `234b5a00` passes Linux native 42/42 and the strict Vulkan suite 25/25, Windows native 43/43 and all 25 strict CPU components, plus Radeon VII MoE/qwen35. Both native device gates execute the capability assertion. Windows initially passed 23 components; paired published/candidate probes confirmed sandbox output-path denial, and only CLI/roundtrip were rerun with normal access and passed. Original failures remain retained. Platform subcases are reported separately: Windows short-path aliases are skipped on Linux, and POSIX file-size-limit injection on Windows. Earlier focused checks cover 491 model checks/388 exact refusals and 397 qwen35 architecture checks, including CPU mixed acceptance, streamed prompt/follow-up identity, down-only differences and missing gate/up text.
- **Identity and timing:** against original base `46f60b64`, all 24 byte captures and records match across CPU/device on dense Qwen3, both synthetic routed families and real Qwen3-30B-A3B Q4_K_M. All 99 compiled shaders match. The [checkpoint report](benchmarks/expert-types-20261002.md) and [data](benchmarks/expert-types-20261002.json) retain all 24 monitored timing calls and a same-owner layout control. Most cells are near flat; the apparent real-MoE CPU decode gain also occurs in the unchanged-behavior control and does not establish a speedup. No material slowdown is demonstrated in this round. The reference-runtime comparison remains a phase-level gate; no mx parity is claimed. No kernel or shader arithmetic changes.
- **Build identity:** `.tmp-expert-types-windows-20261002/build/Release/llmx.exe` is the fresh clean-source gate build `llmx 0.1.0+g234b5a00c5b2`, SHA256 `68626a99a6480973030b850c91b1372a1e5f885a892eec20af7f7546982a205f`. The older worktree-local `build-fixed/Release/llmx.exe` remains the dirty development build `0.1.0+g8db68602c92c.dirty`; it is not the completed gate binary. The rig's source/binary/model identities and final cleanup audit are retained beside the report's evidence.
- **Integration:** rebased onto published main `fb366b16`; only this status file conflicted and both records were kept. The source/test/build range-diff is unchanged for the failing-test and repair commits. Fresh clean builds at `47232075` pass Windows 43/43 native tests, Linux 42/42, and the Windows docs/dead-code/architecture checks. All 99 Linux shader modules match the original candidate. The original model suites, identities and timing keep their recorded source scope under the conflict-free rebase rule. The final amendment changes only this record and its evidence.
- **Landing:** this feature lands by fast-forward after hosted CI passes at its exact head and both main tips are verified unchanged. Local gates are complete; hosted CI and the coordinated landing remain. If main moves first, refresh the integration as AGENTS.md requires.
- **Gotchas:** the down projection is a separate call and may use another type. The helper does not validate tensors or throw when either is missing. Existing refusal fixtures retain their order; combined defects follow the established capability-before-resolution precedence. The qwen35 mixed-tag plan fixture checks planning only, not execution of those altered tiny views.

## Dtype release documentation correction (2026-10-02)

- **Done:** corrected the remaining private-branch and unadvertised-F16 claims against the released CPU and Vulkan capability declarations. The retained publication audit confirms all seven jobs in [CI run 36987806876](https://github.com/mxxm-t/llmx/actions/runs/36987806876) passed at `8af97e88`, then GitHub and Gitea main advanced to that exact commit. The published Windows executable reports `llmx 0.1.0+g8af97e888262`; the docs and source dead-code checks pass against it.
- **Landing:** rebased onto the gated storage main `0b2b0f20`, preserving its record and the split-hold record. This docs-only correction lands by fast-forward after its `docs` and `dead-code` checks; no runtime source changes. The following dtype entry retains its pre-publication checkpoint, with the completed release recorded above.
- **Gotchas:** this changes no runtime policy, qualification result or performance claim. Historical measurements and open reference-speed work remain intact; native F16 does not mean literal half arithmetic on every path or native BF16 support.

## Storage metadata independent of execution support (2026-10-02, branch refactor/storage-types-20261002)

- **Done:** `quant/types.hpp` owns the names, block layouts and checked row sizing of 35 active GGML storage types. Unknown and retired IDs are refused with the tensor's name. The decoder registry reads that table and still implements the same eight types. GGUF metadata and CLI inspection no longer need a decoder. Model construction still refuses a weight its assigned backend cannot run and an unused tensor no model backend supports, before adoption; raw conversion checks every decoder before mapping or allocating decoded payloads.
- **Done:** the layouts were checked against declarations in mx-llama.cpp `eefc4e7321c869496146697d63362f073941aed6` (`ggml.h`, `ggml.c` and `ggml-common.h`), without copying implementations. The native GGUF test covers 345 independent binary cases and compares all eight type IDs and eleven declared block constants in `q.glsl` with the C++ owner. CLI fixtures inspect and tokenize Q2_K files while inference and conversion refuse the unsupported tensor and preserve existing output files. Shader expressions and inference arithmetic are unchanged.
- **Checkpoint checks:** fresh MSVC CPU build; GGUF validation, backend errors and model validation pass, 3/3 (the last includes 484 checks and 385 recorded refusals). The `cli`, `roundtrip`, `f32`, `docs` and `dead-code` components pass. Tiny F32 HF maximum logit error is 0.00000072 at the 2e-5 bound; maximum NLL error is 0.00000290 at 1e-5. Qwen3-0.6B Q8_0 gives the published `8af97e88` executable's three complete top-32 logit outputs and 32 greedy IDs and text on the CPU. This functional comparison uses different build configurations and is not timing evidence.
- **Evidence:** local `.tmp-storage-types-run-20261002/` holds the clean compile log and commands under `final-focused/`, and pinned-model/binary hashes, commands and stdout under `identity/`; the native log is the branch build's `Testing/Temporary/LastTest.log`. The first greedy harness compared timing lines too and reported failure; its preserved generated text already matched, and the corrected check uses the suite's parser and explicit token IDs. A sandbox filesystem refusal in the initial CLI run was followed by the passing run with normal filesystem access; neither attempt is presented as a runtime regression.
- **Completed-branch correctness:** rebased cleanly onto main `46f60b64`. At `8923c376`, the clean Windows CPU build passes 38 native tests and all 25 Python components, requiring the pinned models and tools. The Linux Vulkan build passes 42 native tests and all 25 MI50 components without a reported skip. Six pinned models on both CPU and MI50 give main's exact full-F32 batched, per-token and 64-token greedy captures: 12 cases and 36 binary comparisons. All 99 compiled shader modules are byte-identical between the two arms. The Linux records are under `/zpool1/llmx-xdev-validation/storage-types-20261002/`; the Windows records remain in the local evidence directory above.
- **Timing assessment:** all eight initial calls and six CPU control calls are retained, with activity monitoring. Qwen3-0.6B Q8_0 CPU prefill is below main in all four paired comparisons: -1.11, -3.81, -3.60 and -1.23 percent. GPU prefill's initial paired median is +0.02 percent, GPU decode -1.76 percent. Full emitted-code inspection shows the candidate and info-only control have identical benchmark, model and backend function bytes, yet candidate/control prefill differs by -6.01 and -0.52 percent and decode by +6.21 and +8.04 percent. Accept the timing screen at this limited repeatability: it cannot distinguish the approximately 2.4 percent main comparison from run/initial-state variation. This establishes no hot-code layout band, exact nonregression or speed gain; all negative measurements remain in the report.
- **Gate evidence:** [the report](benchmarks/storage-types-20261002.md) and its companion JSON preserve every timing call, source and binary identities, independent correctness results, activity observations and diagnostic limitations.
- **Landing:** this feature lands by fast-forward after hosted CI passes at its exact head and main is verified unchanged. The local and rig correctness results above are complete; the final record changes only documentation from the measured candidate. F16/BF16 weight execution is still separate, and the dispatch/kernel-class parts of quantization step 0 remain open.
- **Gotchas:** metadata recognition does not claim a decoder, a writer or backend support. F16/BF16 weight execution remains a separate feature with the quantization plan's exact widened-F32 identity requirement. This branch does not mark all of step 0 complete: its older dispatch-table and kernel-class requirements must be reconciled with the already landed precision, load-time refusal and exact-resume owners. No arithmetic or shader expression changes are intended here.
- **Integration refresh:** rebased onto the independently gated split-hold main `58a696ce`. Code merged automatically; this status file retained both new entries. The source/test/build patch is unchanged. Fresh builds at `ae20c5d6` pass 38/38 Windows native tests and 42/42 Linux/MI50 native tests; all 99 shaders match the prior candidate. The final amendment changes only this record and its evidence. Exact-head hosted CI remains required before fast-forward; the completed numerical and timing campaigns above retain their original source identities, as the conflict-free code rebase rule permits.


## Split decode at one card's speed (2026-10-02, branch perf/split-hold, lands by fast-forward)

- **Cause:** on a split each card waits while the other runs its stage. llmx waited on the host, so a waiting card read as idle and both stayed at 930 MHz through a split decode, against 1606 to 1725 MHz on one card; the reference waits on the device, which reads as busy, and its card held 1725 MHz. Both at the automatic performance level, nothing set (`rocm-smi` read only).
- **Done:** `Backend::hold_between_submissions`, which the model asks of each device when its roles use more than one. On Vulkan each `submit()` is followed by a submission that waits on an event, set by the next submission or, after 100 ms without one, by a watchdog thread the first hold starts, so an idle card still idles and no wait nears the driver's job timeout; the backend's own flushes (a chunk, an upload, a read, `sync`) set a pending event and add none. The CPU ignores it. A model asks once it is made and gives the request back when it goes, the backend holding while any request remains. `placement` checks the devices asked and the requests' balance, `backend-vulkan` copies around holds and after an idle gap, `vulkan-lifetime` a hold setup failing at each step and then made again.
- **Gates** (each tree built from its own sha, default clocks, GPU[2] and GPU[3] of the MI50 machine):
  - Decode tg256 at depth 512, arms main `8af97e88`, change, change, main:

    | model | one MI50, main | one MI50, change | two MI50s, main | two MI50s, change |
    |---|---|---|---|---|
    | Qwen3-8B Q8_0 | 70.75, 70.66 | 70.46, 70.58 | 47.51, 45.74 | 69.01, 67.57 |
    | Qwen3.6-27B Q8_0 | 22.60, 22.62 | 22.58, 22.70 | 19.12, 19.22 | 22.59, 22.67 |

  - The reference on the same two cards, Qwen3-8B Q8_0, 512 tokens: 70.3 tok/s on one MI50 and 61.7 split.
  - `llmx-split-check` on Qwen3-8B Q8_0 over the two MI50s bit-identical to one; CTest 42 of 42 on an MI50; on the Radeon VII, CTest 43 of 43 under the AMD proprietary driver and Qwen3-8B Q8_0's ids main's on `vulkan:0` and on `vulkan:0,cpu` at shares 3,1, where the card holds while the CPU runs its layers.
  - Qwen3-8B and Qwen3.6-27B Q8_0's 64 greedy ids over the two MI50s are main's.
  - Through a 400-token split `generate` both cards held 1725 MHz at 96 to 100 percent busy, and both were at 930 MHz and idle within half a second of its end, the watchdog letting the last holds go.
  - Qwen3-8B Q8_0 served over the two MI50s (`tools/server_load.py`, 256-token prompts, 128-token replies, arms main, change, change, main), output tok/s and inter-token p50: one user 44.1 and 42.9 at 20.0 and 20.7 ms on main against 59.2 and 59.6 at 14.3 and 14.2 ms; 4 users 164.4 and 163.6 against 164.8 and 165.2; 16 users 279.9 and 279.3 against 278.5 and 277.7; 32 users 279.6 and 279.9 against 275.1 and 278.2: with passes in flight both cards are busy already, and the holds cost nothing beyond the runs' spread.
  - The device suite on an MI50 passes every component it runs but `dead-code`, which found the hold state's struct name unused (now unnamed), and `perf`, whose prefill floor the first synthetic bench in a fresh container misses on main as on the branch (568 and 387 tok/s against 1000 while the idle card raises its clock, then 1870 to 2300 on both), where the synthetic model runs on one device and takes no hold.
  - Rebased onto main `46f60b64` without a conflict, so the builds, CTest, `docs`, `dead-code` and the hosted run ran again at the head.
  - The other developer's review found three lifecycle defects, each checked against the code and fixed: the test's last copy freed its buffers before it retired, a hold setup failing part way left its resources and a retry overwrote them, and a model enabled holds for good before its construction could still fail. The fixes reran CTest on an MI50 and the Radeon VII, the split timing and the hosted run. The watchdog's release is held by the clocks above, not by a test, since every submission also releases a hold.


## Activation dtype across CPU and Vulkan (2026-10-02, pre-publication checkpoint)

`--dtype auto|f16|bf16|f32` resolves once during placement and reaches every model command, the server and each execution stage. Current CPU and Vulkan backends prefer qualified F16 under auto; explicit F32 remains available, and BF16 emulation or a wider fallback is reported. Weight storage, KV storage and the retained F32 state operations are separate. Vulkan MXFP4 support lands with the completed policy.
- **Done:** independent HF depth comparisons are complete on CPU and MI50 for all three frozen 16k histories, against both original-F32 quality and the preselected literal-F16 reference. Each covers all five reference logits at all 512 scored positions. CPU raw16 has 511 exact top-one matches, one accepted tie and no miss; CPU raw8/chat and all three MI50 histories have 512 exact matches. The accepted tie's HF gap is 0.002174377 under the frozen 0.1 rule. Retained qualification passes include Windows 25 Python components and 43 native tests, Linux 42 native tests, MI50 AUTO 25 components and all 90 BF16 physical split cases. The test-first CPU fit dependency (`33965fa9`, then `d34ded17`) is integrated at `c7cb1407`; native F16 preference and its reviewed documentation are applied.
- **Landing:** one dtype commit follows the separately committed CPU fit regression and repair; the stack lands by fast-forward after the hosted workflow at its exact head passes and main is verified unchanged. Linux CPU has all 25 components covered, with raw-blocks and integrated server recovery passing beside the earlier unaffected numerical evidence. The final 32-call timing is complete, retaining all four matched blocks and activity observations; prior six-thread results remain separately scoped.
- **CI repair:** hosted run [36984465433](https://github.com/mxxm-t/llmx/actions/runs/36984465433) built the Vulkan backend and passed its native and CPU suite steps, then rejected `Backend::take_matrix_paths` as reachable only from tests. The unused getter is removed; direct backend tests read the same evidence through the existing stage-swap operation in `tests/matrix_precision.hpp`. Model reporting, dispatched kernels and numerical bounds are unchanged. Focused native and linked reachability checks cover this cleanup; the amended landing head still requires its own hosted pass. No finding is allowlisted.
- **Gotchas:** the earlier server failure reproduced on main under the same 8 GiB limit and led to the CPU fit repair below; a larger cap alone would not establish that repair. Final timing uses matched builds and the same 512-token prompt and 128-token decode workloads. The llmx benchmark disables automatic KV fitting, so the reserve cannot shrink its cache; its three arms have identical capacity. The reference uses a smaller reserved context, which must be distinguished from the matched executed histories. Header and host/server changes can move hot code, so previous numerical or timing results do not replace the final integrated timing. Native F16 preference is applied, not still withheld; this does not claim native BF16 hardware arithmetic or literal half arithmetic on every F16 path.


The [final CPU report](benchmarks/dtype-cpu-final-20261002.md) and [data](benchmarks/dtype-cpu-final-20261002.json) compare main, candidate, a reporter-site layout control and mx in one environment. Rates are median tokens/s; changes are medians of within-block percentages, not ratios of the median rates.

| Model | Phase | Main | Dtype | Control | mx target | Dtype vs main | Dtype vs mx |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | pp512 | 13.550 | 16.555 | 16.215 | 20.390 | +19.12% | -18.66% |
| Qwen3-8B Q4_K_M | tg128 | 5.345 | 5.840 | 5.085 | 6.160 | +1.43% | -12.18% |
| Qwen3-0.6B Q5_K_M | pp512 | 186.735 | 191.430 | 177.145 | 111.330 | -4.39% | +73.27% |
| Qwen3-0.6B Q5_K_M | tg128 | 44.175 | 46.820 | 44.330 | 48.470 | +5.69% | -3.89% |

**Performance assessment:** accept the workload tradeoff under AGENTS.md: Q4 prefill improves in all four blocks and Q5 decode improves in all four; Q5 prefill loses in three, with a median paired cost of 4.39 percent. This is not universal nonregression or reference parity. Activity is uneven: main has 82 busy-CPU intervals of 620, candidate 2 of 534; main's third Q4 call alone accounts for 81 busy intervals. All 2178 monitor samples and 1002 missed process observations remain, with no gap over 2.5 seconds. The aggregate gains are not clean kernel-effect estimates, and the reporter control bounds only its CLI site. All 32 calls returned zero, all executable hashes and the actual four-CPU quota were verified at the end, and the owned container is stopped.

**Open speed work:** the current CPU Q4 prompt/decode and Q5 decode reference gaps above remain open, as do the separately scoped CPU Q8/MXFP4 and GPU reference cells retained in the history. The pinned MI50 assessment retains its accepted Q4 decode precision cost and its MXFP4 prefill gap; it is not relabeled as a timing of this final source. Direct activation packing and reference-informed CPU tiling, plus Vulkan MXFP4 prompt work, follow on separate branches. No numerical bound is weakened for speed, and no unvalidated optimization is included here.

[Historical dtype checkpoints and execution contract](benchmarks/dtype-execution-history-20261002.md) preserve all prior rows, failed attempts, source identities, scoped performance results and evidence locations. [PRECISION](PRECISION.md) defines the current policy and approved historical delivery plan.

The integrated archive has SHA256 `e66d21c7cdbdc6fc332dca878d4fcc8a48aab7f06a57d63ca36c7a61ec6fbf4f`; its build and six affected native checks pass (dtype, host-memory, placement and the three server checks). The Linux executable is `llmx 0.1.0+unknown`, SHA256 `714d5ab7e8b440e31256149e2589daa0376f8ccbe794d0c96390062c11444850`, under `/zpool1/llmx-xdev-validation/dtype-native-f16-integration-20261002`. That archived build has no Git metadata. Startup, the native checks and the first real-model server's 66 requests/prefix checks ran under the original 8 GiB cap without an OOM. After that server exited, the container's cap was raised in place to 12 GiB for the remaining functional checks; no case was restarted, and source, executable, commands and precision bounds are unchanged. The complete component therefore has mixed 8/12 GiB coverage, not a full-suite 8 GiB claim. Unrelated CPU activity is recorded and elapsed time is not performance evidence. Of the 251 compiled, test and build files, 245 still match the frozen archive byte for byte. The sole production delta removes the unused two-line backend getter above; five test files move its calls to the test helper, with three wrapped comments also joined to follow the comment convention. The source audit reconstructs exactly these changes against the archive; all other runtime, test and build files remain unchanged. The final documentation review reconciles the preceding 83-page review with the promoted policy, its owner pages and the newly separated historical record; unchanged pages retain that review, and no new full-source audit is claimed.

The 8 GiB phase exposed heavy reclaim/refault churn, recorded in `server-resource-window.json`: a ten-second window had 17.99 seconds of system CPU time, 0.94 seconds of user CPU time, about 1.21 million file refaults and 1.52 million direct scans, with negligible quota throttling and no increase in physical read bytes. This prompted the capacity amendment, recorded in `memory-cap-amendment.json`; a proposed 16 GiB amendment was deferred and never applied. The reserve fixes the demonstrated startup OOM, but does not establish good throughput under this pressure. A separate placement follow-up should account for the unique host-read mapped-weight working set before maximizing backed KV, while handling cold/warm AUTO, mapped/direct loading and shared CPU budgets. Core's general reclaimable-cache estimate and backend scratch are separate concerns; no such follow-up implementation is included here.

The full server component now passes without skips, including the six-turn conversation, 640-token re-prefill reuse matching a fresh server, unrelated donor, departed-client and EOS cases. Its container and supervisor are terminal; all source and executable hashes were verified again. The audit is `completion-audit.json` in that evidence directory, SHA256 `1f3456f0b6c86aba189bdde74c418dfd18b547d01b0171049634d3ffa6b09075`; it retains 3491 memory samples, peak 10236342272 bytes and zero OOM events or kills. That peak belongs to the mixed-cap functional run, not the separate 8 GiB startup result.

The first timing launch found that the control-build one-CPU quota had not been cleared by an update to zero. Its preflight failed, but the orchestration erroneously continued and started one main Q4 call. The owned container was stopped before a second call; the partial attempt and its monitoring are preserved as an invalid resource configuration, with no rate used in the final comparison. The corrected launch uses an explicit four-CPU quota and a hard successful-preflight guard before the unchanged 32-call comparison starts.

The documentation, dead-code and architecture-boundary checks pass after that reconciliation. The first check command used the nonexistent component name `architecture` and stopped at argument parsing; the corrected `arch-boundary` command ran all three checks. This harness typo did not launch or replace a runtime measurement.

## CPU fit leaves room beyond payload buffers (2026-10-02, lands by fast-forward)

- Main `25549f02` has the runtime of `0bf4fcae`. Its Qwen3-0.6B Q8_0 server and the dtype qualification both fitted 36736 F32 KV tokens under a fresh 8 GiB cgroup and were killed before health. This is a preexisting fit defect, not a dtype numerical regression: the fit left no backend reserve for page tables, allocator overhead and CPU workspaces. The recorded page tables alone reached about 16.1 MiB.
- The existing CPU `scratch_reserve` now returns one twentieth of reported free memory, and the existing layer fit counts each host backend's reserve as it already counted copying backends' reserves. Explicit buffer sizes and inference arithmetic are unchanged.
- The host-reserve regression failed before the fix. After it, placement, host-memory, server-resume, server-passes and server-passes-cpu passed, 5/5. The same startup under the same 8 GiB limit reached healthy with 34816 KV tokens, a measured peak of 8162009088 bytes and zero cgroup OOM events; the smaller fitted pool is the capacity cost of leaving operating room.
- Evidence: `/zpool1/llmx-xdev-validation/dtype-cpu-fit-oom-20261002`, including the failed original attempts, source hashes, build logs, 100 ms memory samples and health records. Fixed binary SHA256 `e97f62b637de5a186c7317ea2d619f73942db946a1b3b237b25773590975f9bc`, version `llmx 0.1.0+unknown`; the archived build has no Git metadata. The initial main command's unsupported `--dtype` refusal and a nonexistent build-target harness error are retained separately from the valid before/after comparison.
- Throughput is not inferred from these startup checks. The integrated release refresh builds the performance arms alike and gives each llmx arm identical capacity and executed histories with automatic KV fitting disabled, so this reserve cannot change its benchmark capacity. The reference's reserved context is smaller and is reported separately from the matched histories. Independent HF results remain applicable because no arithmetic changed.
## The embedded MTP proposer for qwen35 (2026-10-02, branch feat/qwen35-mtp, step 4 of SPECULATIVE, lands by fast-forward)

- **Done:** the code on the CPU and Vulkan with its tests and docs; the drafter's ops on both; the fit's count of the drafter; `bench --model --drafter embedded` (the k = 0 gate and the rollback lines). Its gates and measurements, in the order they were taken:
  - The block's math: the tiny fixture's drafts and every draft row's logits at steps 1 and 2, from prompts of 1, 2, 5 and 12 tokens, against the assembled HF reference (`tools/gen_baseline.py qwen35-mtp`, docs/ASSETS.md): every pick and drafted id equal, largest logit error 3.0e-7 on the CPU and 5.8e-7 on the Radeon VII against the F32 bound of 2e-5.
  - Identity, drafts on against off, ids compared, `generate --verbose`, 128 tokens, greedy on a copy prompt and a chat story, and seeded at temperature 0.8: equal in every cell on one MI50 for Qwen3.6-27B-MTP Q8_0, Qwen3.8-27B Q8_0 and Qwen3.6-27B-MTP Q4_1 at 1 and 3 drafts, and over two MI50s (layer split) for both Q8_0 files; the tiny fixtures on the Radeon VII and the CPU; on Windows CTest 41 of 41 and the CPU suite.
  - k = 0 (the block loaded, drafting nothing), Qwen3.6-27B-MTP Q8_0 on one MI50, `bench --model`, arms off, embedded, embedded, off: pp512 262.27, 260.23 off against 260.36, 260.43 (-0.3 percent), tg128 22.15, 21.19 against 21.68, 21.40 (-0.6 percent), pp16384 205.24 against 204.79 (-0.2 percent).
  - Rollback, the same file on one MI50, after a mark and a verify of 3 drafts, each rejection position timed as completed work up to the next step's logits, medians of 10 runs against keeping all 4 rows (a step of about 44 ms): rerun copying the saved inputs back, +2.56, +2.74 and +2.80 ms (5.8 to 6.4 percent); reading them where they were saved, +1.99 to +2.23 ms (4.5 to 5.0 percent); with each phase of every state layer recorded unordered (`Backend::unordered`), +1.09 to +1.14 ms (2.5 to 2.6 percent). The gate is 2 percent: not met yet.
  - The retract's own call, timed apart (e9305710, GPU[7], cores 4 to 7): 0.37 to 0.66 ms of a rollback of +0.81 to +0.94 ms (1.8 to 2.1 percent of a 44.2 ms step) after a 512-token prompt, and 0.47 to 0.60 ms of +0.89 to +1.02 ms (1.9 to 2.2 percent) after 4096 tokens, so recording the rerun's 96 dispatches on the host is more than half of it; on GPU[6] and cores 8 to 11 the same code measured 2.5 percent. Submitting the rerun in chunks, so the device starts while the host records the rest, loses: after a 512-token prompt, arms interleaved on GPU[7] and cores 4 to 7, one submission (e9305710) cost 2.35 to 2.44 percent, a submission every 24 state layers 2.36 to 2.47, every 12 2.68 to 2.92 and every 6 2.82 to 3.07, the retract's own call growing by about 30 us a submission. A second run of every 24 layers measured a 46 ms step and is flagged: a build ran on cores 8 to 11 beside it. The experiment trees are not kept.
  - Where the rest goes (an experiment build of 8a5163a7 timing the rerun's dispatches, not kept): the rerun spans 1.02 to 1.12 ms on the device with every dispatch timestamped, its 48 delta rule dispatches 37 to 40 us each and overlapping about twice, the driver giving that kernel 84 registers and 18.9 KB of shared memory, 3 subgroups a SIMD; recording it takes the host 0.24 to 0.49 ms and submitting it 0.12 to 0.23 ms. So the device is the larger part. The delta rule's short build (staging 4 tokens a block for calls whose views hold at most 8 rows, a column's arithmetic unchanged) measured, arms 8a5163a7, short, short, 8a5163a7 on GPU[7] and cores 4 to 7: a rollback of +0.81 to +1.03 ms (1.83 to 2.35 percent of a 44.0 ms step) against +0.96 to +1.07 ms (2.17 to 2.40 percent of 44.3 to 44.5 ms), tg32 23.02 and 23.00 tok/s against 22.84 and 22.25, pp512 257.48 and 257.37 against 256.96 and 256.41; the per-token logits of a prompt and 128 greedy ids with 3 drafts were the same bytes from both builds, CTest passed 40 of 40 on the MI50 and the Vulkan, qwen35 and spec CTests on the Radeon VII, and the qwen35 and decode-probe components on the MI50. Still at the gate's edge: the register count is next.
  - The delta rule reading its state once the first block is staged (eab3daa3) takes the short build from 84 registers to 48, 5 subgroups a SIMD, and the 16-token build to 72; a subtle change to the short build's rounding alone, within the CPU bound, fails backend-vulkan's 8- and 9-row comparison, so the test holds the two builds together. From 14:50 the arms were disturbed by the cards' own heat at default clocks: GPU[7] idles at 43 C, GPU[1] reached a junction of 104 C at its 225 W cap and ran 1032 to 1606 MHz, the step growing from 44 to 47 to 50 ms, so those arms (rollback -4 to +20 percent) are flagged and not counted. Started once GPU[6]'s junction is below 55 C, still at default clocks, four arms gave a step of 43.80 to 44.10 ms and a rollback of +0.56 to +1.14 ms, 1.28 to 2.61 percent, median 2.0: at the gate, not under it with a margin.
  - The rollback measured as the median over repeats of each position's difference from keeping all 4 in the same repeat, so a clock drifting with the card's heat moves both (`bench`, 60f48305), 20 repeats, GPU[6], each arm started below 55 C, default clocks: the rerun as it is, two submissions, +0.71 to +0.83 ms, 1.62 to 1.89 percent of a 43.7 ms step, middle halves within +0.58 to +1.83 ms, in two arms (a third, whose step read 47.6 ms, flagged); the rerun as one submission, no chunk submitted while barriers are left out, +0.89 to +1.17 ms, 2.04 to 2.67 percent, in three. So the rerun stays as it is, under the 2 percent gate at every rejection position on one MI50; the split's is measured on the landing head.
  - The gates at 415865a2, rebased on main fb366b16 (split-hold and the storage metadata among it), on cores 4 to 7, each timed arm started below 55 C at default clocks: CTest 42 of 42 on an MI50 and 43 of 43 on the Radeon VII, where the qwen35, decode-probe, f32 and moe components pass on the device; the suite on the CPU and on an MI50 passes but raw-blocks, which needs numpy the container lacks, and on the MI50 perf, whose synthetic prefill floor main misses on the same card too (544.8 and 535.4 tok/s against 1000); the Qwen3-0.6B and Qwen3.5-0.8B identity cells the same bytes as main's, 14 of 14 on the CPU and 14 of 14 on the MI50; drafts on against off on Qwen3.6-27B-MTP Q8_0, greedy and seeded, the same ids on one MI50 and on two. A timing round against main with drafts off, arms main, branch, branch, main on GPU[6]: the 27B's pp512 261.60, 260.93 against 261.27, 260.83 tok/s, tg128 21.38, 21.77 against 21.89, 20.96, pp16384 203.90, 203.92 against 204.06, 203.80; Qwen3-0.6B Q8_0 pp512 8702, 8606 against 8646, 8559 (within their run-to-run spread of 120 to 200), tg128 392.71, 390.81 against 391.18, 391.87. The rollback, paired: one MI50 1.67 to 1.79 percent, middle halves +0.63 to +0.88 ms (a second arm, its step read 46.3 ms, flagged); a two-MI50 split 0.52 to 1.22 and 0.62 to 1.40 percent of a 44.5 to 45.0 ms step. The hosted run's Vulkan job found `Model::drafts`, which only a test called, and it went (22f95978).
  - At 3f6257f3, rebased on main 2c293678 with the drafter's dtype fix: CTest 42 of 42 on an MI50 and 38 of 38 on the CPU on Windows, the linked dead-code check, drafts on against off on Qwen3.6-27B-MTP Q8_0 the same ids greedy and seeded, and the Qwen3-0.6B and Qwen3.5-0.8B identity cells the same bytes as main's, 14 of 14 on the CPU and on the MI50. The rollback, paired, each arm started below 55 C: over two MI50s 0.02 to 0.43 and 0.50 to 0.63 percent of a 44.4 to 45.2 ms step; on one MI50 1.62 to 1.89 percent, middle halves +0.59 to +0.89 ms (a second arm, its step read 46.7 ms, flagged). Under the 2 percent gate at every rejection position on both.
  - What the rerun adds to the architecture contract, after the coordinator's review: two plan fields and one step field, no hook. `LayerPlan::recur_writes` names the slots an update writes, so the rerun gives each layer a room of its own and runs every layer's update with no barrier between layers; `LayerPlan::recur_phases` (2 on qwen35: the conv, then the recurrence that reads its output) and `Step::phase` let it run one phase of every layer, then the next. Together they took the rollback from 4.5 to 5.0 percent of a step to 2.5. The rerun reads the mark's saved inputs in place by giving its calls the mark's rows as `Step::rows`, so no field says where a plane starts. Per-position draft counts moved into `spec::Acceptance::verified(kept, fed)`, and `blocks::nextn_input` builds its interleave order in a plain local.
  - BOSS's comparison against the frozen reference (mx-llama.cpp f2a54df595, built for gfx906 in `mixa3607/rocm-gfx906:7.2.1-complete` with one fix to a call that did not compile, its Q5_1 routed product, a path these files do not reach; `llama-server -m M -ngl 99 -fa on -c 20480 -np 1 -v [--spec-type draft-mtp --spec-draft-n-max D] [-sm layer -ts 1,1 or 1,1,1]`, its batch settings the defaults, n_batch 2048 and n_ubatch 512, and every layer split's log saying `pipeline parallelism enabled`; `/completion` with the same raw text, the one user message in the Qwen chat frame, greedy, the end token ignored, `cache_prompt` false; its acceptance from its per-verify `accepted a/n draft tokens` lines). llmx: `generate --file F -n N --temp 0 --ignore-eos --drafter off|embedded --draft-max D --device ...` at llmx 415865a2, its ubatch the default 512. Same prompts, same depth on both sides, arms interleaved by depth, timing on cores 4 to 7; each row is one prompt at one draft depth, and the equality columns compare a runtime's text with drafts against its own without.
  - Measured at 415865a2, rebased on main fb366b16 with split-hold, which stand for the landing head: what came after changes nothing the generate path runs on an MI50, the host tier being the server's alone, the mixed expert fix a refusal at load, the dispatch refactor the same kernels by another table, and the drafter's activation dtype fix leaving f16, the device's effective dtype, as it was; drafts on against off and the rollback are measured at 3f6257f3 and after. Each runtime's block of cells started once its cards were below 55 C, default clocks; the load reached 46 during the runs. The earlier tables at 9a512b02, before split-hold, are in this block's history.
  - One MI50 (GPU[1]):
  | model | prompt, tokens | depth | llmx tok/s | ref tok/s | llmx vs ref | llmx kept/drafted by position | ref kept/drafted by position | llmx = off | ref = off |
  |---|---|---|---:|---:|---:|---|---|---|---|
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | off | 21.16 | 22.07 | -4.1% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 1 | 31.17 | 29.28 | +6.5% | 120/134 | 120/134 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 2 | 38.99 | 35.88 | +8.7% | 85/96 73/96 | 85/96 73/96 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 3 | 43.57 | 38.31 | +13.7% | 69/79 57/79 50/79 | 68/78 56/78 49/78 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 4 | 40.88 | 40.87 | +0.0% | 59/66 50/66 43/66 37/66 | 59/65 51/65 42/65 35/65 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | off | 22.22 | 22.09 | +0.6% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 1 | 31.45 | 28.90 | +8.8% | 118/137 | 117/136 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 2 | 38.03 | 35.55 | +7.0% | 87/99 69/98 | 87/98 68/98 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 3 | 40.17 | 34.04 | +18.0% | 71/86 55/85 43/84 | 69/86 55/86 43/85 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 4 | 35.10 | 37.14 | -5.5% | 61/75 51/74 40/73 28/73 | 59/73 51/73 40/72 30/72 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | off | 21.58 | 22.09 | -2.3% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 1 | 31.67 | 30.01 | +5.5% | 123/131 | 123/131 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 2 | 40.16 | 37.73 | +6.4% | 87/93 75/92 | 86/92 75/92 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 3 | 46.74 | 40.75 | +14.7% | 69/73 61/72 52/72 | 68/72 61/72 52/72 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 4 | 41.38 | 42.25 | -2.1% | 61/63 50/62 44/62 37/61 | 60/64 49/64 42/64 38/63 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | off | 17.27 | 18.12 | -4.7% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 1 | 23.72 | 24.47 | -3.1% | 111/127 | 117/136 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 2 | 27.27 | 32.73 | -16.7% | 87/100 67/99 | 85/99 70/99 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 3 | 32.55 | 31.35 | +3.8% | 71/85 53/84 45/83 | 72/82 55/82 43/82 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 4 | 28.99 | 34.25 | -15.4% | 62/74 47/73 41/72 30/72 | 62/73 48/73 39/73 32/73 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 2048 | off | 18.75 | 19.11 | -1.9% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | explain 2048 | 1 | 27.28 | 25.84 | +5.6% | 892/1011 | 969/1076 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 2048 | 2 | 35.15 | 36.48 | -3.6% | 681/794 572/794 | 688/757 601/757 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 2048 | 3 | 41.05 | 39.11 | +5.0% | 558/665 459/665 365/664 | 547/635 467/635 394/635 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 2048 | 4 | 35.36 | 36.10 | -2.0% | 488/586 397/586 319/586 257/586 | 501/601 397/601 303/601 244/601 | yes | no |
  | Qwen3.8-27B-Q8_0 | copy 256 | off | 20.29 | 22.08 | -8.1% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | copy 256 | 1 | 33.62 | 29.05 | +15.7% | 121/133 | 121/133 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 2 | 38.12 | 36.24 | +5.2% | 84/95 76/95 | 83/93 76/93 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 3 | 44.48 | 40.11 | +10.9% | 69/77 61/77 48/76 | 68/76 60/76 48/76 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 4 | 39.09 | 39.46 | -0.9% | 61/68 54/68 39/67 33/67 | 60/67 54/67 38/67 33/67 | yes | yes |
  | Qwen3.8-27B-Q8_0 | explain 256 | off | 21.17 | 22.05 | -4.0% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | explain 256 | 1 | 29.65 | 28.53 | +3.9% | 112/143 | 115/138 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 2 | 31.72 | 31.76 | -0.1% | 82/111 61/111 | 80/110 63/110 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 3 | 35.28 | 34.96 | +0.9% | 72/93 50/93 39/93 | 71/89 54/89 38/89 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 4 | 31.59 | 35.35 | -10.6% | 65/85 45/85 34/85 25/84 | 68/77 52/77 35/77 21/77 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | off | 20.89 | 21.76 | -4.0% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | code 256 | 1 | 32.08 | 29.14 | +10.1% | 120/134 | 119/135 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | 2 | 36.58 | 36.79 | -0.6% | 86/96 72/96 | 88/95 71/95 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | 3 | 40.07 | 40.33 | -0.6% | 71/81 62/81 41/80 | 71/77 61/77 43/77 | yes | yes |
  | Qwen3.8-27B-Q8_0 | code 256 | 4 | 37.80 | 39.95 | -5.4% | 59/71 51/71 42/70 32/70 | 59/68 53/68 39/68 33/67 | yes | yes |
  | Qwen3.8-27B-Q8_0 | long16k 256 | off | 16.45 | 20.41 | -19.4% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 1 | 27.11 | 27.59 | -1.8% | 118/137 | 119/134 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 2 | 29.68 | 33.38 | -11.1% | 83/99 72/99 | 86/97 71/97 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 3 | 32.63 | 33.56 | -2.8% | 69/86 57/85 43/85 | 69/87 55/87 43/87 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 4 | 28.73 | 36.18 | -20.6% | 62/77 51/77 37/76 28/76 | 61/69 52/69 40/69 30/69 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 2048 | off | 19.70 | 20.93 | -5.9% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | explain 2048 | 1 | 30.54 | 29.47 | +3.6% | 877/994 | 974/1072 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 2048 | 2 | 35.74 | 35.69 | +0.2% | 678/800 568/800 | 678/779 588/779 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 2048 | 3 | 41.26 | 40.63 | +1.6% | 557/665 452/664 373/664 | 553/612 469/612 409/612 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 2048 | 4 | 36.67 | 44.39 | -17.4% | 482/589 389/589 319/589 267/589 | 452/490 408/490 364/490 331/490 | yes | no |
  - Two MI50s, a layer split (GPU[6] and GPU[7]), every reference server logging `pipeline parallelism enabled`:
  | model | prompt, tokens | depth | llmx tok/s | ref tok/s | llmx vs ref | llmx kept/drafted by position | ref kept/drafted by position | llmx = off | ref = off |
  |---|---|---|---:|---:|---:|---|---|---|---|
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | off | 22.67 | 20.76 | +9.2% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 1 | 34.21 | 28.07 | +21.9% | 120/134 | 120/134 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 2 | 39.59 | 33.65 | +17.7% | 85/96 73/96 | 85/96 73/96 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 3 | 44.42 | 35.66 | +24.6% | 69/79 57/79 50/79 | 68/78 56/78 49/78 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 4 | 41.34 | 40.04 | +3.3% | 59/66 50/66 43/66 37/66 | 59/65 51/65 42/65 35/65 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | off | 22.77 | 21.23 | +7.3% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 1 | 34.11 | 27.47 | +24.2% | 118/137 | 117/136 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 2 | 38.63 | 32.05 | +20.5% | 87/99 69/98 | 87/98 68/98 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 3 | 40.57 | 35.19 | +15.3% | 71/86 55/85 43/84 | 69/86 55/86 43/85 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 4 | 36.70 | 36.87 | -0.5% | 61/75 51/74 40/73 28/73 | 59/73 51/73 40/72 30/72 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | off | 22.59 | 21.37 | +5.7% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 1 | 35.14 | 28.61 | +22.8% | 123/131 | 123/131 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 2 | 40.95 | 36.47 | +12.3% | 87/93 75/92 | 86/92 75/92 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 3 | 47.36 | 42.17 | +12.3% | 69/73 61/72 52/72 | 68/72 61/72 52/72 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 4 | 43.06 | 41.85 | +2.9% | 61/63 50/62 44/62 37/61 | 60/64 49/64 42/64 38/63 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | off | 20.55 | 19.45 | +5.7% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 1 | 28.62 | 26.12 | +9.6% | 111/127 | 117/136 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 2 | 31.61 | 29.55 | +7.0% | 87/100 67/99 | 85/99 70/99 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 3 | 33.26 | 34.99 | -4.9% | 71/85 53/84 45/83 | 72/82 55/82 43/82 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 4 | 29.24 | 33.93 | -13.8% | 62/74 47/73 41/72 30/72 | 62/73 48/73 39/73 32/73 | yes | no |
  | Qwen3.8-27B-Q8_0 | copy 256 | off | 22.72 | 21.67 | +4.8% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | copy 256 | 1 | 35.01 | 29.08 | +20.4% | 121/133 | 121/133 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 2 | 40.55 | 36.02 | +12.6% | 84/95 76/95 | 83/93 76/93 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 3 | 45.18 | 40.06 | +12.8% | 69/77 61/77 48/76 | 68/76 60/76 48/76 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 4 | 39.83 | 38.19 | +4.3% | 61/68 54/68 39/67 33/67 | 60/67 54/67 38/67 33/67 | yes | yes |
  | Qwen3.8-27B-Q8_0 | explain 256 | off | 22.77 | 21.55 | +5.7% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | explain 256 | 1 | 32.44 | 28.01 | +15.8% | 112/143 | 115/138 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 2 | 34.53 | 31.21 | +10.7% | 82/111 61/111 | 80/110 63/110 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 3 | 37.53 | 34.44 | +9.0% | 72/93 50/93 39/93 | 71/89 54/89 38/89 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 4 | 32.19 | 25.07 | +28.4% | 65/85 45/85 34/85 25/84 | 68/77 52/77 35/77 21/77 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | off | 22.71 | 21.62 | +5.1% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | code 256 | 1 | 34.72 | 28.60 | +21.4% | 120/134 | 119/135 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | 2 | 39.65 | 36.25 | +9.4% | 86/96 72/96 | 88/95 71/95 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | 3 | 41.81 | 39.35 | +6.3% | 71/81 62/81 41/80 | 71/77 61/77 43/77 | yes | yes |
  | Qwen3.8-27B-Q8_0 | code 256 | 4 | 38.40 | 38.69 | -0.7% | 59/71 51/71 42/70 32/70 | 59/68 53/68 39/68 33/67 | yes | yes |
  | Qwen3.8-27B-Q8_0 | long16k 256 | off | 20.55 | 19.88 | +3.4% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 1 | 29.46 | 26.90 | +9.5% | 118/137 | 119/134 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 2 | 32.17 | 31.97 | +0.6% | 83/99 72/99 | 86/97 71/97 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 3 | 32.91 | 33.12 | -0.6% | 69/86 57/85 43/85 | 69/87 55/87 43/87 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 4 | 27.55 | 35.56 | -22.5% | 62/77 51/77 37/76 28/76 | 61/69 52/69 40/69 30/69 | yes | no |
  - Three MI50s, a layer split (GPU[1], GPU[6] and GPU[7]), every reference server logging `pipeline parallelism enabled`:
  | model | prompt, tokens | depth | llmx tok/s | ref tok/s | llmx vs ref | llmx kept/drafted by position | ref kept/drafted by position | llmx = off | ref = off |
  |---|---|---|---:|---:|---:|---|---|---|---|
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | off | 21.98 | 21.49 | +2.3% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 1 | 32.57 | 28.82 | +13.0% | 120/134 | 120/134 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 2 | 38.10 | 34.52 | +10.4% | 85/96 73/96 | 85/96 73/96 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 3 | 42.05 | 37.50 | +12.1% | 69/79 57/79 50/79 | 68/78 56/78 49/78 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | copy 256 | 4 | 40.28 | 38.47 | +4.7% | 59/66 50/66 43/66 37/66 | 59/65 51/65 42/65 35/65 | yes | no |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | off | 22.05 | 21.55 | +2.3% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 1 | 32.53 | 28.58 | +13.8% | 118/137 | 117/136 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 2 | 37.41 | 34.38 | +8.8% | 87/99 69/98 | 87/98 68/98 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 3 | 39.19 | 34.32 | +14.2% | 71/86 55/85 43/84 | 69/86 55/86 43/85 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | explain 256 | 4 | 35.85 | 37.27 | -3.8% | 61/75 51/74 40/73 28/73 | 59/73 51/73 40/72 30/72 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | off | 22.15 | 21.52 | +2.9% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 1 | 33.71 | 29.77 | +13.2% | 123/131 | 123/131 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 2 | 39.39 | 36.80 | +7.0% | 87/93 75/92 | 86/92 75/92 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 3 | 45.99 | 43.06 | +6.8% | 69/73 61/72 52/72 | 68/72 61/72 52/72 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | code 256 | 4 | 42.24 | 42.28 | -0.1% | 61/63 50/62 44/62 37/61 | 60/64 49/64 42/64 38/63 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | off | 20.31 | 19.88 | +2.1% |  |  |  |  |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 1 | 26.87 | 26.67 | +0.7% | 111/127 | 117/136 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 2 | 29.49 | 31.89 | -7.5% | 87/100 67/99 | 85/99 70/99 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 3 | 31.87 | 33.70 | -5.4% | 71/85 53/84 45/83 | 72/82 55/82 43/82 | yes | yes |
  | Qwen3.6-27B-MTP-Q8_0 | long16k 256 | 4 | 28.36 | 34.24 | -17.2% | 62/74 47/73 41/72 30/72 | 62/73 48/73 39/73 32/73 | yes | no |
  | Qwen3.8-27B-Q8_0 | copy 256 | off | 22.53 | 21.50 | +4.8% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | copy 256 | 1 | 33.09 | 27.36 | +20.9% | 121/133 | 121/133 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 2 | 38.78 | 36.25 | +7.0% | 84/95 76/95 | 83/93 76/93 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 3 | 43.48 | 40.62 | +7.0% | 69/77 61/77 48/76 | 68/76 60/76 48/76 | yes | yes |
  | Qwen3.8-27B-Q8_0 | copy 256 | 4 | 38.83 | 38.12 | +1.9% | 61/68 54/68 39/67 33/67 | 60/67 54/67 38/67 33/67 | yes | yes |
  | Qwen3.8-27B-Q8_0 | explain 256 | off | 22.64 | 21.51 | +5.3% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | explain 256 | 1 | 31.29 | 26.34 | +18.8% | 112/143 | 115/138 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 2 | 33.01 | 31.25 | +5.6% | 82/111 61/111 | 80/110 63/110 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 3 | 36.32 | 34.92 | +4.0% | 72/93 50/93 39/93 | 71/89 54/89 38/89 | yes | no |
  | Qwen3.8-27B-Q8_0 | explain 256 | 4 | 31.24 | 32.75 | -4.6% | 65/85 45/85 34/85 25/84 | 68/77 52/77 35/77 21/77 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | off | 22.53 | 21.31 | +5.7% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | code 256 | 1 | 32.93 | 26.77 | +23.0% | 120/134 | 119/135 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | 2 | 37.55 | 36.31 | +3.4% | 86/96 72/96 | 88/95 71/95 | yes | no |
  | Qwen3.8-27B-Q8_0 | code 256 | 3 | 41.28 | 40.16 | +2.8% | 71/81 62/81 41/80 | 71/77 61/77 43/77 | yes | yes |
  | Qwen3.8-27B-Q8_0 | code 256 | 4 | 36.92 | 37.72 | -2.1% | 59/71 51/71 42/70 32/70 | 59/68 53/68 39/68 33/67 | yes | yes |
  | Qwen3.8-27B-Q8_0 | long16k 256 | off | 20.40 | 19.78 | +3.1% |  |  |  |  |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 1 | 27.42 | 26.01 | +5.4% | 118/137 | 119/134 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 2 | 31.04 | 33.11 | -6.2% | 83/99 72/99 | 86/97 71/97 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 3 | 31.74 | 33.37 | -4.9% | 69/86 57/85 43/85 | 69/87 55/87 43/87 | yes | no |
  | Qwen3.8-27B-Q8_0 | long16k 256 | 4 | 27.53 | 36.12 | -23.8% | 62/77 51/77 37/76 28/76 | 61/69 52/69 40/69 30/69 | yes | no |
  - Read at the same depth, llmx against the reference. One MI50: Qwen3.6-27B-MTP Q8_0 off -4.7 to +0.6 percent, d1 +5.5 to +8.8 (-3.1 on the 16k prompt), d2 +6.4 to +8.7 (-16.7 on 16k, -3.6 at 2048 tokens), d3 +5.0 to +18.0, d4 -5.5 to 0.0 (-15.4 on 16k); Qwen3.8-27B Q8_0 off -8.1 to -4.0 (-19.4 on 16k), d1 +3.6 to +15.7, d2 -0.6 to +5.2 (-11.1 on 16k), d3 -0.6 to +10.9, d4 -17.4 to -0.9 (-20.6 on 16k).
  - Two MI50s, with split-hold: off +3.4 to +9.2 percent, d1 +9.5 to +24.2, d2 +0.6 to +20.5, d3 -4.9 to +24.6, d4 -22.5 to +28.4, the 16k prompt the low end of each depth from d3 on. Three MI50s: off +2.1 to +5.7, d1 +0.7 to +23.0, d2 -7.5 to +10.4, d3 -5.4 to +14.2, d4 -23.8 to +4.7, again lowest on 16k.
  - llmx's text with drafts equals its text without in every cell; the reference's differs in the 2048-token cells and in most Qwen3.8 cells, so its acceptance there is counted on another text, and where both texts are equal acceptance by position is within a few drafts. The 16k cells are the verify's attention, each row reading the whole history (the follow-up below); llmx's d4 against d3, where the reference holds level, is not yet explained.
- **Follow-up, owned by the coordinator, outside this step:** a verify's 2 to 5 rows of one sequence each take the per-row attention kernel, which reads the whole history once a row, so at 16k tokens the KV read grows with the depth, which matches llmx falling from 3 percent behind the reference without drafts to 7 to 14 percent behind with them on the 16k prompt (not yet measured). The coordinator is building the per-row attention build that reads K and V once for every row of one view, each row's arithmetic unchanged; perf/attention-three-heads (ece55048), three query heads a workgroup where the GQA group divides by three, as the 27B's six do, is in test beside it.
- **Landing:** one commit on main 5835886c, after the coordinator's review (the drafter's products dropping the activation dtype, fixed with a witness test) and XDEV's (`embed_ids` decoding an MXFP4 table as Q6_K, fixed with its own build and every embedded type checked against `embed`; a failed rerun leaving its device recording unordered, fixed and held by `spec`'s injected failure; bench's rollback reaching past a short context, now skipped with its reason and held at the edge by the `qwen35` component; the dtype, already fixed), and a hosted run green at its head; it lands by fast-forward. At e2555324, before the last rebase, CTest passed 42 of 42 on an MI50 and the Vulkan CTests on the Radeon VII, with the qwen35 and decode-probe components on the MI50. Next is step 5, `feat/spec-server`.
- **Gotchas:** the host tier (`Model::save_host`, `restore_host`) copies a history's KV, the drafter's layer among it, and its checkpoint slot, but not the drafter's carried row, which sits beside the state slots on the head's device; the server loads no drafter until step 5, which must carry that row with the slot so a restored history drafts as one never evicted.

## One row class: a generated token's row and a prompt's row the same bits (2026-10-01, investigation, no branch yet)

- **First measurement** (`exp/one-row-class`, never merged: the row-class check reporting, for four rows of one prompt, how many outputs at extent 1 differ from extents 2, 64, 512 and 1000):
  - MI50: extents 1 and 2 give the same bits everywhere (both on the row kernels). From the tile crossover on, F32 and Q8_0 matmuls and the routed F32 and Q8_0 products differ by summation order only (largest relative 2.3e-4); Q4_0, Q4_1, Q4_K, Q5_K and Q6_K differ by up to 158 percent relative on small outputs, since the decode row kernels read the 8-bit twin and the tile the 16-bit one; attention differs by order only (6.1e-4) between the per-row kernel and the tile.
  - CPU: attention gives the same bits at every extent; every matmul type differs from extent 2 on, F32 and Q8_0 by order (8.6e-4), the K-quants by up to 24 percent through the 8-bit decode dots, the routed products by order (1.9e-4).
- **So:** the precision part (8-bit decode activations on both backends) is the dtype plan's steps 4 and 5 (docs/PRECISION.md), which move those rows to 16 bits; what remains after it is summation order in the matmul's float block sums and in attention's reduction. The next measurement waits for those steps: the dispatch and arithmetic audit of each op, tails, grouped and routed calls, the head and residual paths, attention's lengths, F32 exceptions and dtype in the probe's identity, then the cost of one order on each path.
- **Rule:** no row class merges until every op of every stage, CPU and streamed experts included, gives identical rows over the claimed extents and the end-to-end reuse checks pass; if one order costs too much, the classes and 2c stay. No gate is weakened by it.

## Decode attention takes three heads a workgroup (2026-10-02, branch perf/attention-three-heads, lands by fast-forward)

- **Done:** once a row's history fills every part, a decode attention workgroup takes four, three or two heads of a KV head as the group divides, where it took four or two; each head's arithmetic is the same in either grouping. `backend-vulkan`'s attention after a long history adds query/KV ratios 3 and 6, which take the three-head path, to 1, 2, 4 and 8.
- **Gates** (each tree built from its own sha, default clocks, one MI50):
  - Decode on Qwen3.6-27B Q8_0, tg256, arms main, change, change, main: after 16384 tokens 20.09 and 20.05 tok/s on main `25549f02`'s kernels against 20.57 and 20.52; after 4096 tokens 21.76 and 21.72 against 21.83 and 21.90. Models whose group divides by four take the build they took.
  - Qwen3.6-27B Q8_0's 64 greedy ids after a ~4k-token prompt are main's.
  - On an MI50, CTest 40 of 40 and the device suite with `--require-tools` passing every component it runs (the qwen35 gate fixtures not on that disk skip, as MXFP4 does on the device); on the Radeon VII, CTest 41 of 41 and Qwen3-8B Q8_0's ids after a ~4k-token prompt main's.
  - The hosted run at the head. Rebased onto main `8af97e88` without a conflict, so the builds, CTest and the hosted run ran again there.

## The attention merge reads each part's state once (2026-10-01, branch perf/attention-merge, lands by fast-forward)

- **Done:** each of a row and head's parts has its maximum and sum read once into shared memory and its weight formed once, where each of 128 lanes read every part's state and formed every weight; the maximum is taken across subgroups, which is exact in any order; a lane loads eight parts' values of its column before it sums them, in part order, as one lane reading every part did, so the output keeps its bits. The parts a row may take are at most 256, the merge's workgroup (`attention_split_max`).
- **Gates** (each tree built from its own sha, default clocks):
  - The merge's device time over those 4096 dispatches, at 64 parts: 13.2 ms on main, 7.5 ms with the change. Decode after 16384 tokens, tg256, on one MI50:

    | model | base | change |
    |---|---|---|
    | Qwen3-30B-A3B Q4_K_M, at 64 parts (main `87051ea3`) | 57.35, 57.89 | 60.28, 60.55 |
    | Qwen3-30B-A3B Q4_K_M, at 32 parts (main `59760e1e`) | 61.19, 60.89 | 61.77, 61.52 |
    | Qwen3-30B-A3B Q4_K_M, after 4096 tokens, at 32 parts | 84.47, 89.75 | 92.95, 91.51 |
    | Qwen3-8B Q8_0, at 32 parts | 48.54, 49.60 | 48.99, 50.06 |

  - The same greedy ids as main after a ~4k-token prompt on Qwen3-8B Q8_0, Qwen3-30B-A3B Q4_K_M and Qwen3.5-0.8B Q4_K_M (heads 256 wide) on an MI50, and on Qwen3-8B Q8_0 on the Radeon VII.
  - On an MI50, CTest 39 of 39 and the device suite with `--require-tools` passing every component it runs (the qwen35 fixtures not on that disk skip, as MXFP4 does on the device); on the Radeon VII, CTest 40 of 40, and decode after 4096 tokens on Qwen3-8B Q8_0 at 36.87 and 36.17 tok/s against main's 36.25 and 35.00.
  - The hosted run at the head.
- **Standing after it:** Qwen3-30B-A3B Q4_K_M after 16384 tokens stands at about 62 tok/s against the reference's 69.5; Qwen3-8B Q8_0 at about 50 against 43.7 and Qwen3-0.6B Q8_0 at about 160 against 159.4. The rest of the gap is the per-row attention kernel's own time, 45 percent of decode at that depth.
- **Gotchas, measured on the way and not kept** (`exp/` branches on Gitea, never merged):
  - Heads per workgroup and the parts cap, swept with knobs read from the environment in an experiment build: on Qwen3-30B-A3B, Qwen3-8B and Qwen3-0.6B Q8_0 after 16384 tokens the profile's numbers (four heads of a KV head or two, 32 parts) were the fastest of 1, 2 and 4 heads by 16, 32, 64 and 128 parts.
  - The four-head build taking one token at a time rather than two: 72 registers rather than 76, still three subgroups a SIMD, and 1 to 3 percent slower.
  - Key and value rows kept as their packed halves until each value is used: 64 registers and four subgroups a SIMD, but 53.6 against 62.6 tok/s on Qwen3-30B-A3B, the unpacking repeated for each head costing more than the subgroups gained.
  - A whole KV group's query heads in one workgroup, with a tile of 32 tokens' keys and values staged once in shared memory for all of them (`exp/attention-tile-lds`): after 16384 tokens 54.7 against 62.6 tok/s on Qwen3-30B-A3B Q4_K_M at its best (eight heads or four, 64 parts), 41.1 against 44.3 on Qwen3-8B Q8_0 and 132.3 against 161.6 on Qwen3-0.6B Q8_0; it also broke batch invariance, its tokens' lanes following the heads a workgroup took, which the dispatch chooses from the batch's longest row. The kernel is bound by its arithmetic and occupancy rather than its loads; whether the F16 policy lets attention take f16 operands with F32 sums is asked of the dtype work.
  - The online softmax's rescale skipped where a token leaves the running maximum unchanged, so its factor is exactly 1 and the bits stay main's (`exp/attention-lazy-rescale`, 2026-10-02): Qwen3-8B Q8_0's full logits rows over 1100 tokens, batched, per token and 64 greedy steps, and Qwen3.6-27B Q8_0's greedy ids after a 4k prompt main's, but no gain beyond the runs' spread on one MI50 (after 16384 tokens 20.15 and 18.45 tok/s on the 27B against main's 20.33 and 20.41, 48.74 and 50.78 on the 8B against 49.70 and 45.00), the kernel's speed following its occupancy rather than its multiplies.

## Decode attention at depth: fewer parts on an MI50 (2026-10-01, branch perf/attention-splits, lands by fast-forward)

- **Done:**
  - The cap on a row's attention parts is a profile number per head width: `attention_split_max` for heads 128 wide and narrower, 32 in the MI50 row and 64 by default, and `attention_split_max_wide` for heads 256 wide, 64.
  - On an MI50 the gain is the merge's: `--profile` of decode after 16384 tokens gave `attention_merge` 13.2 ms at 64 parts and 7.7 ms at 32 on Qwen3-30B-A3B Q4_K_M (17.7 and 10.1 ms on Qwen3-8B Q8_0) of 4096 dispatches, while the attention kernel took the same time.
  - Heads 256 wide stay at 64 parts: at 32 their lane groups, half as many a workgroup, take 64 tokens each, and Qwen3.6-27B Q8_0 fell from 19.9 to 18.5 tok/s at depth, every sample lower.
  - The Radeon VII keeps 64: at 32, Qwen3-8B Q8_0 decode after 4096 tokens gave 19.6 and 19.8 tok/s against main's 27.8 and 34.4.
- **Gates** (each tree built from its own sha, default clocks):
  - MI50 timing against main `ef33cf78`, two rounds of base, fix, reference, reference, fix, base on one card, with 32 parts for every head:

    | model | cell | main | branch | reference |
    |---|---|---:|---:|---:|
    | Qwen3-30B-A3B Q4_K_M | tg512 @ d16384 | 56.41 | 59.66 (+5.8%) | 69.51 |
    | Qwen3-30B-A3B Q4_K_M | tg128 | 128.46 | 130.87 | 105.86 |
    | Qwen3-30B-A3B Q4_K_M | pp512 | 1118.18 | 1119.02 | 1073.25 |
    | Qwen3-8B Q8_0 | tg512 @ d16384 | 46.63 | 48.55 (+4.1%) | 43.74 |
    | Qwen3-8B Q8_0 | tg128 | 73.03 | 72.78 | 56.35 |
    | Qwen3-8B Q8_0 | pp512 | 831.37 | 830.48 | 859.01 |

    Prompt cells are level, as prompts take attention's tiled kernel. With heads 256 wide at 64 parts, Qwen3.6-27B Q8_0 at tg512 after 16384 tokens gave 19.96 and 20.01 tok/s on main `87051ea3` and 19.92 and 19.97 on the branch, arms main, branch, branch, main, and Qwen3.5-0.8B Q4_K_M's 64 greedy ids after a 3688-token prompt were main's.
  - `tools/long_context_check.py` on Qwen3-8B Q8_0 through `serve` on an MI50: two fresh servers gave the same 512 tokens, and the CPU took each as its top choice at 512 of 512 positions, the largest gap 0.000 logits.
  - On an MI50, CTest 39 of 39 and the device suite with `--require-tools` passing every component it runs (the qwen35 fixtures not on that disk skip, as MXFP4 does on the device); Qwen3-8B Q8_0 `logits` on three short prompts gave main's bytes, since their histories split as before.
  - The Radeon VII under Windows: CTest 40 of 40; at 32 parts the 16k long-context check on Qwen3-0.6B Q8_0 passed at 512 of 512, the largest gap 0.000; at the head, where it keeps main's 64, Qwen3-8B Q8_0's 64 greedy ids after a ~4k-token prompt were main's, and decode after 4096 tokens gave 34.35 and 34.19 tok/s against main's 34.79 and 33.99, the arms interleaved while the other developer's Windows tests, started at 07:55 without timing, ran on the same PC.
  - The hosted run at the head.
- **Gotchas:** the per-row kernel's speed follows its occupancy, not its arithmetic. A tiled softmax (`perf/attention-tiled`, parked) scored 16 tokens a lane group with a transposed butterfly, keeping the scores' bits, and updated the softmax once a tile; it took 84 registers where main's kernel takes 40 (six subgroups a SIMD to three) and 132 in the four-head build where main's takes 76 (three to one), and decode after 16384 tokens fell from 51.3 to 34.6 tok/s on Qwen3-30B-A3B Q4_K_M and from 42.6 to 33.1 on Qwen3-8B Q8_0. On the Radeon VII, `bench --depth 16384` on Qwen3-8B Q8_0 stops with "bad allocation" on main as on the branch.

## A restarted server waits for the card to give back its predecessor's memory (2026-10-01, lands as `fix/fit-settle`)

- **Found:** in production a Qwen3.8-27B Q8_0 server restarted on the card of an old 27B server was refused twice before an automatic restart about 13 s later loaded. Sampled every 0.1 s, an MI50 gave back an ended 27B server's 32 GB in steps over about three seconds, holding it level for more than two seconds between steps, while the fit gave up after two quiet reads (0.5 s).
- **Done:** `settle` waits five quiet seconds (`kSettleQuiet`, 20 reads of 250 ms), up to thirty in all, before a fit short of what it asked for stands; `3e53ebfe`, a device whose free memory stays level for four seconds before rising, fails without it.
  - Restarting a Qwen3.6-27B Q8_0 server with SIGTERM and starting a Qwen3.8-27B Q8_0 server on the same MI50 at once, in one container: five of five loaded with the change, one of three with the two-quiet-read settle of 8d, one of three before it.
  - CTest 35 of 35 on the CPU build and 39 of 39 on an MI50 (`placement` given 120 s, since its refusals and cut budgets each wait the five quiet seconds), and the CPU suite with `--require-tools` passing every component.
- **Gotchas:** a start whose budget the cards cannot hold now takes five seconds more, the price of telling a card still giving memory back from one that will not.

## The Windows CI job's limit (2026-10-01, branch fix/windows-ci-budget, lands by fast-forward)

- **Why:** the Windows CPU job ran 13 min 54 s to 14 min 52 s on main's last three hosted runs against its 15-minute limit, and twice ended at the limit on feat/qwen35-checkpoints with every step passed (runs 36843077334 and 36844790789), the build alone taking 2 min 46 s to 3 min 51 s by runner.
- **Done:** the limit is 20 minutes on Windows; Linux keeps 15 and macOS 25 (`docs/CI.md`).

## Donors kept in host memory, written back and promoted (2026-10-02, branch feat/host-cache, step 2b part a of SPECULATIVE, lands by fast-forward)

- **Done:** the code, the docs (SERVER, USAGE, `docs/src/model-history.md`, `docs/src/server.md`) and the tests: `kv-cache` and `arch-qwen35` round-trip a history through host memory into other blocks and slots, bit for bit, with the refusals; `server-resume` alternates two conversations on a pool that holds one, on the synthetic Q8_0 model and the hybrid one with a checkpoint slot, one CPU and two, every follow-up promoting its donor and forking 256 tokens with its reply on a fresh model, and none without the tier, and injects a failing copy at write-back and at promotion; the `server` component does the same on Qwen3-0.6B Q8_0 against the CLI.
- **Gates** at `db4d95a8` on the rig against main `25549f02`, the MI50 GPU 5 and cores 12 to 15: CTest 40 of 40; the suite on the CPU passes every component and on the MI50 every one, MXFP4 skipped as on main; Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the MI50; the linked dead-code check passes once the `BlockKVStorage::block_tokens` line, which the copies now read, leaves `tests/data/known_findings.txt`. Two conversations alternating on a 1024-token pool, each follow-up promoted from host memory, give `generate`'s ids on its prompt and reuse 448 to 512 tokens of 574 to 627 on Qwen3-0.6B and Qwen3.5-0.8B Q8_0 on one MI50 and over the MI50 and the CPU, and on Qwen3-30B-A3B Q4_K_M with its experts on the CPU and prompts from 32 tokens streamed.
- **Measured**, Qwen3.8-27B Q8_0 on one MI50 at default clocks, `--max-seqs 8 --ctx-size 32768`, which keeps 3 checkpoint slots: six chat conversations without reasoning taken in turn for five turns each, as six chat UI users would, about 2k tokens each by the end, then the same with other text beside an unrelated streaming request; five arms interleaved, main, the change with `--host-cache-bytes 8589934592`, the reference server at its defaults, the change, main; the llmx replies the same bytes in every arm; load average 8 to 10, the MTP agent timing on other cards and cores:

  | follow-up turns (24 a run) | main | change | reference |
  |---|---:|---:|---:|
  | time to first token p50, alone | 7.28, 7.28 s | 2.27, 2.26 s | 2.27 s |
  | tokens read in all, of 40220 | 40220, 40220 | 10908, 10908 | 9862 of 40141 |
  | time to first token p50, beside a stream | 6.84, 6.82 s | 2.36, 2.35 s | 2.95 s |
  | the stream's gap between tokens p50 / p90 / p99 / max | 54.0 / 369 / 2114 / 2180, 53.2 / 368 / 2113 / 2180 ms | 53.4 / 58.9 / 1601 / 2081, 53.1 / 54.1 / 1584 / 2082 ms | 59.7 / 60.9 / 1269 / 2011 ms |

  On main every follow-up read its whole conversation, since 6 conversations and their re-read replies outran 3 checkpoint slots; with the host tier 41 follow-ups of 48 were promoted (`host_hits`), 27 entries and 8.5 GB held at the end and 37.6 GB copied either way over the two runs. The stall the copies put on the scheduler thread, from the server's own line per copy, is the copy's enqueue: a promotion 0.6 to 1.2 ms, a write-back 1.0 ms at p50 and 114 to 119 ms at p90, the tail being the slabs' first allocation, about 30 ms a 64 MiB slab, which reuse ends once the tier has reached its cap (one write-back of 821 ms among 220). A first build that read the copies to the host on the scheduler thread took 257 ms a write-back at p50 and 2.2 s at most, 0.84 GB/s; enqueued into host-visible memory allocated per copy, 95 ms at p50, the allocation itself. The unrelated stream's gaps show no stall: its p90 and p99 fall below main's, since the follow-ups' prompt passes are a quarter as long, and its largest gap is main's.
- **Gates after the review's fixes**, at 5621e9ff on main 46f60b64: CTest 42 of 42, the linked dead-code check, the suite on the CPU with every component passing and on the MI50 the components the change touches (`server`, `qwen35`, `chat`, `f32`, `moe`, `split`), the real-model host probes as before, and the hosted run (36998683388) green on all seven jobs; the recheck's sizing fix reruns CTest and the hosted run.
- **Coordinator's review** of e11cdc2a: the default was read before the model loaded, and a copy was refused only below its own size, so the tier could take the headroom the CPU fit keeps and bring back the out-of-memory the fit's reserve prevents on a memory-limited host; and with every cache on the CPU the tier copied host memory into more host memory. The default is now read after the load and is none where every cache is on the CPU, and the copy's new slabs must leave the CPU's reserve, the rule held at its edge in `kv-cache`. XDEV's glance found the reserve read before the idle slabs the copy frees were freed, which refused copies those slabs would have made room for (confirmed; it is read after them).
- **Left:** landing, once the rebase onto main 58a696ce has its builds, CTest and hosted run. The slabs' first allocation stays on the scheduler thread; a server that should not take it can be given a smaller `--host-cache-bytes`, and moving it off the thread waits for a measured need.
- **Gotchas:** part (b), the message-boundary checkpoints, is not in this branch, so an edited earlier message reads its history again (3368 tokens in step 2c's measurement), as on main. On the hybrid model a follow-up's donor ends at its inherited checkpoint, the same history as the entry it was promoted from, which `write_back` renews rather than copying again. The cap counts whole slabs, so a short history takes a slab a device.

## A reply read again as prompt rows, idle and while it is written (2026-10-01, branch feat/idle-reprefill, step 2c of SPECULATIVE)

- **Done:** the code, the docs (SERVER, USAGE's health fields, `docs/src/server.md`, `docs/src/inference-chat.md`) and the tests: `server-resume` reads a 300-token prompt's reply again idle and while it is written, on the synthetic Q8_0 model and the hybrid one with checkpoints, on one CPU and over two, and the follow-up forks every whole block of the next turn's ids (384 and 640 tokens against 256 without the job) and gives its reply on a fresh model; the hybrid job is cancelled by a request at each of its seven pass boundaries, the request giving its reply alone and the job completing afterwards; idle, a regenerated reply forks the request's own donor at the prompt's 256 tokens beside the job's donor and gives the same reply; the writing case leaves nothing to read once the reply has ended, which caught a job made whole after its last pass never completing; a follow-up answered at once, before the job has read the rest, forks the running job's 384 tokens; and a request that fits beside the job leaves it running. `chat-template` holds the new overload to its renders, a template that renders reasoning only for the last turn included. The `server` component's real-model pass adds a chat follow-up on Qwen3-0.6B Q8_0 that reuses past its first turn's prompt (640 of 704 tokens against a 531-token first prompt on the CPU) with the reply of a fresh server.
- **Gates so far**, on the rig's MI50 (GPU 5) and its CPU, against step 3 (d1455eee): CTest 40 of 40 (`server-resume`'s answered-at-once case over a two-CPU split failed once under load and was relaxed for passes in flight, then passed four runs in a row); the suite on the CPU passes every component and on the MI50 every one but `perf`'s prefill floor (701 against 1000 tok/s at load 8 to 13, the synthetic bench, which the change does not touch), `server` passing its new chat follow-up on both (640 of 704 tokens reused against a 531-token first turn); Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give step 3's bytes in all 14 identity cells on the CPU and the MI50.
- **Follow-ups against `generate`**, a ~600-token chat turn without reasoning, its 200-token reply read again, then a follow-up of 48 greedy tokens: on Qwen3-0.6B Q8_0 and Qwen3.5-0.8B Q8_0 on one MI50, over the MI50 and the CPU as a layer split, and on the CPU alone, and on Qwen3-30B-A3B Q4_K_M with its experts on the CPU and prompts from 32 tokens streamed to the MI50, the follow-up reused 768 of 831 or 843 tokens, and its ids equal the same request on a fresh server and `generate` on its rendered prompt, every cell.
- **The use**, Qwen3.8-27B Q8_0 on one MI50 at default clocks, `--max-seqs 8 --ctx-size 32768`, a 14-turn chat without reasoning posted to `/v1/chat/completions` as a chat UI sends it, and an edited sixth turn; turns 2 to 13, five arms interleaved (the change, step 3, the reference server, step 3, the change), the llmx arms' replies the same bytes in every arm:

  | | step 3 | change | reference |
  |---|---:|---:|---:|
  | turns back to back: time to first token p50 | 2.74, 2.77 s | 2.51, 2.51 s | 1.79 s |
  | turns back to back: tokens read a turn, p50 | 581.5 | 459 | 444.5 |
  | 3 s between turns: time to first token p50 | 2.75, 2.77 s | 2.25, 2.27 s | 1.82 s |
  | turn 13 (7461 tokens), 3 s apart | 3.63 s, 677 read | 2.88 s, 549 read | 2.01 s, 543 read |
  | beside an unrelated stream: time to first token p50 | 2.92, 2.93 s | 2.85, 2.86 s | 2.14 s |
  | beside it: tokens read a turn, p50 | 581.5 | 517.5 | 444.5 |
  | the unrelated stream's inter-token gap p50 / p90 / p99 | 54.2 / 58.1 / 2086, 54.3 / 57.6 / 2086 ms | 53.9 / 56.1 / 1686, 53.8 / 55.3 / 1687 ms | 59.9 / 61.2 / 1349 ms |

  A follow-up now reads its new message and at most a block beside it, as the reference does, so the time to first token falls by 0.25 to 0.5 s a turn on these 400-token messages, more where a reply is longer: the production case of a 9370-token read after a long reply reads only the new message. The remaining gap to the reference is the prompt read itself (llmx reads the same ~450 tokens about 0.5 s slower on the MI50, measured in 8c). Beside a stream the change read 517.5 tokens a turn rather than 459, since the whole next-turn ids reached the scheduler only after the client had already sent the follow-up; they now go out before the reply's last chunk (a traced rerun after that change: every follow-up from turn 10 forked the job's whole ids and read only its new message). The 64-row chunks (`kJobChunk`) move the unrelated stream's p50 and p90 gaps by under 2 ms, and its p99, set by the follow-ups' prompt passes, falls with the shorter reads. The edited turn reads all 3368 tokens on both llmx arms; the message-boundary checkpoints of 2b part (b) are its fix.
- **The renders' cost:** a chat route renders and tokenizes a Qwen3.8-27B conversation in 4.2, 15.2 and 30.2 ms (p50) at 2033, 7722 and 16580 tokens on the rig's CPU, through `/v1/tokenize`; `Api::next_turn` renders twice, every 32 tokens while a reply is written (about 2 s at 17 tok/s, so 1.6 to 3 percent of a core a conversation, on its connection thread) and once before the reply's last chunk, which that chunk then waits for.
- **Left:** the hosted run on the rebase onto main 395b4950 (the move of the history operations, no conflict in code); XDEV rechecked the reservation fix at 2a3b4902 with no blocking finding, and the hosted run at 2a3b4902 is the record before the rebase. `server-passes` does not simulate jobs; a job is admitted through `enter` and takes no request's room, so the policy's invariants are unchanged, which the review is asked to confirm.
- **Gotchas:** a job's donor stands beside the one it forked, so a conversation holds two donors and, on a model that keeps a state, two checkpoint slots until donor age takes the older; where the slots are short, the job's checkpoint takes the forked donor's slot and a regenerated reply then recomputes its prompt. The next turn's ids come from the reply as the route returned it, so a client that sends back other text, or on a Qwen 3.8 template omits the `reasoning_content` the route gave, forks only as far as its tokens match. A job's fork counts its shared blocks twice in the ledger, as a request's does, so a job does not start where its source and its own ids do not fit the pool together.

## A sequence's history in one file (2026-10-01, branch refactor/model-history, move only, lands by fast-forward)

- **Done:** `src/model/history.hpp` holds the operations on a sequence's history, `Model::fork`, `reset`, `retract`, `mark`, `keep` and `checkpoint`, and the private steps only they take, `settle`, `rewind`, `saved_at`, `save`, `rerun`, `restore_mark` and `drop_mark`, each body and its comments moved unchanged and defined out of the class; `Model` declares them where they were, and `runtime.hpp` includes the file after the class. `runtime.hpp` goes from 1413 lines to 1220. `docs/src/model-history.md` takes their description from the runtime's page.
- **Gates** (`e3c029bf`, this commit before its gate line): the moved lines are the removed ones, in order, but for the indent and `inline` and `Model::` before each name. On the rig against main `57a8c04a`, each tree built from its own sha: CTest 40 of 40; the suite on the CPU and on an MI50 passes every component but `raw-blocks`, which needs numpy the container lacks, and `mxfp4`, skipped on the MI50; Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the 14 on the MI50; `llmx-split-check` of both, `cpu` against `cpu,cpu` over the excerpt with 8 steps and ubatch 64, bit-identical, its output main's but for the free memory it reads. On Windows, CTest 37 of 37 and the CPU suite. The hosted run 36923198561, green on all 7 jobs; this line alone changed after it.
- **Timing** against main `57a8c04a`, each built from its own detached tree on the rig, `bench --model --threads 4 --r 3` on one MI50 at default clocks (and the CPU, cores 4-7), arms base, change, change, base in two rounds, tok/s in run order; load average 5 to 30 from other work on the machine, recorded per run. No cell moves 2 percent, so no layout control was run.

  | cell | base `57a8c04a` | change `395b4950` | change in the mean, percent |
  |---|---|---|---:|
  | Qwen3-0.6B Q8_0, pp512 | 8540.12, 8552.34, 8553.20, 8544.12 | 8555.08, 8547.38, 8541.65, 8516.54 | -0.09 |
  | Qwen3-0.6B Q8_0, pp4096 | 4431.03, 4429.49, 4430.47, 4430.01 | 4431.99, 4429.80, 4429.44, 4430.35 | +0.00 |
  | Qwen3-0.6B Q8_0, tg128 | 388.79, 391.39, 390.66, 391.25 | 392.67, 391.28, 391.58, 390.92 | +0.28 |
  | Qwen3-8B Q8_0, pp512 | 829.50, 829.30, 829.24, 830.34 | 829.60, 830.55, 829.47, 829.30 | +0.02 |
  | Qwen3-8B Q8_0, pp4096 | 691.39, 691.27, 690.66, 690.54 | 691.34, 690.93, 690.60, 691.19 | +0.01 |
  | Qwen3-8B Q8_0, tg128 | 74.89, 74.76, 74.78, 74.84 | 74.85, 74.86, 74.84, 74.64 | -0.03 |
  | Qwen3.5-0.8B Q8_0, pp512 | 7418.50, 7408.62, 7428.14, 7427.83 | 7419.98, 7424.64, 7416.57, 7420.71 | -0.00 |
  | Qwen3.5-0.8B Q8_0, tg128 | 349.29, 351.21, 347.56, 349.03 | 348.37, 348.62, 351.46, 351.36 | +0.19 |
  | Qwen3.6-27B Q8_0, pp512 | 258.32, 257.70, 258.19, 258.16 | 257.87, 257.65, 257.85, 258.05 | -0.09 |
  | Qwen3.6-27B Q8_0, tg128 | 22.92, 22.93, 22.87, 22.47 | 22.88, 22.88, 22.88, 22.89 | +0.37 |
  | Qwen3-30B-A3B Q4_K_M, tg512 after 16384 | 62.94, 62.26, 63.06, 63.02 | 62.94, 62.99, 61.84, 63.02 | -0.20 |
  | CPU, Qwen3-0.6B Q8_0, pp512 | 222.12, 216.66, 211.15, 215.12 | 215.57, 214.81, 215.27, 218.30 | -0.13 |
  | CPU, Qwen3-0.6B Q8_0, tg128 | 45.94, 46.08, 46.15, 46.24 | 46.06, 46.30, 45.97, 46.14 | +0.03 |

- **Gotchas:** a branch that changed one of these bodies in `runtime.hpp` moves its change to `history.hpp`, where the body sits dedented by one level, `inline` and `Model::` before its name.

## Speculative decoding's verify, with lookup (2026-10-01, branch feat/spec-verify, step 3 of SPECULATIVE)

- **Done:** the code, the docs and the tests: `sampler` (accept against the loop without drafts), the new `spec` CTest (synthetic proposers and lookup through `generate` on dense, routed and hybrid models over one to four CPU stages; the history calls under verifies; an injected rerun failure; the refusals), `llmx-split-check`'s verify phase, and drafts on against off in the `f32`, `moe` and `qwen35` components.
- **First measurement:** Qwen3-0.6B Q8_0 on the Windows machine's CPU, a prompt asking to repeat a list three times, greedy: 160 tokens in 3510 ms without drafts and 1226 ms with lookup at 8 drafts (45.6 and 130.5 tok/s), the same ids; seeded at temperature 0.8 the same ids at `--draft-max` 1, 4 and 16.
- **The acceptance rule, measured:** the coordinator's review found that the first rule, resting 16 tokens after four verifies in a row kept nothing, was the fast adaptation section 3 and lesson 21 rule out; section 3's rule is built instead, one average of drafts kept a verify, moving an eighth of the way each verify from 2, below a break-even resting 16 tokens and then verifying once more. Qwen3-8B Q8_0 and Qwen3.5-9B Q4_K_M on one MI50, 256 greedy tokens, a copy workload (a module copied back with docstrings) and a plain chat (`--chat`, a short story), lookup at 3 drafts, two interleaved rounds, tok/s and drafts verified and kept, the ids the same in every arm; load average 6 to 8:

  | | off | four misses, rest 16 | no rest | average, break-even 1 | average, break-even 0.5 |
  |---|---:|---:|---:|---:|---:|
  | 8B copy | 70.1, 71.6 | 105.0, 107.8 (200 drafted, 117 kept) | 118.1, 118.7 (245, 140) | 114.8, 116.4 (211, 132) | 119.1, 118.7 (245, 140) |
  | 8B chat | 72.7, 73.0 | 69.5, 70.5 (156, 14) | 65.7, 66.0 (323, 20) | 73.1, 73.3 (71, 12) | 72.5, 72.3 (89, 11) |
  | 9B copy | 79.5, 87.9 | 136.0, 141.2 (215, 158) | 145.1, 144.6 (230, 166) | 125.9, 129.5 (206, 142) | 145.0, 144.8 (230, 166) |
  | 9B chat | 88.4, 88.3 | 81.5, 83.2 (168, 34) | 81.6, 81.5 (248, 52) | 85.1, 85.7 (54, 8) | 84.5, 86.3 (150, 38) |

  The off and the earlier arms come from two sessions minutes apart, the off arms repeated in each. At a break-even of half a draft the average keeps all of the copy workload's gain and gives up 0.5 percent of the 8B's plain chat and 2 to 4 percent of the 9B's, where the first rule gave up 3 to 11 percent of the copies' speed against no rest and 4 to 8 percent of the chats' against no drafts; break-even 1 cut the copies' gain. `kBreakEven` is 0.5; the cost model of step 5 is to replace it with a price.
- **Draft length on a device:** Qwen3-8B Q8_0 on one MI50, a base completion copying a Python module back with a docstring added to each function, 256 greedy tokens: without drafts 69.6 tok/s; with lookup at `--draft-max` 1, 3, 7, 8 and 15, 90.9, 103.9, 97.3, 74.3 and 72.1 tok/s. A pass of generated tokens costs by its columns: `bench --model --seqs` gives 75.9 tok/s at 1 sequence, 249.4 at 8, 179.2 at 9 and 254.4 at 16, so a pass of 9 rows takes 50 ms where one of 8 takes 32 and one of 1 13.2. Past 8 rows a verify costs more than its kept drafts save, so `--draft-max` defaults to 3, and pricing a draft by its pass is left to the cost model of step 5.
- **Against the reference's n-gram drafting**, the same prompt on one MI50 at default clocks, two rounds of llmx without drafts, llmx with lookup at its default, mx-llama.cpp's llama-server (build 10951, HIP, one slot, flash attention) without speculation and with `--spec-type ngram-simple` at its defaults, every llmx arm the same ids; load average 9 to 11:

  | tok/s | llmx, off | llmx, lookup | reference, none | reference, ngram-simple |
  |---|---:|---:|---:|---:|
  | Qwen3-8B Q8_0 | 69.7, 70.5 | 104.0, 103.9 | 68.8, 68.6 | 111.8, 112.0 |
  | Qwen3.5-9B Q4_K_M | 79.7, 74.4 | 104.9, 118.2 | 70.7, 70.7 | 73.6, 75.3 |

  On the dense 8B lookup gains 48 percent and stays 7 percent behind the reference's n-gram drafting, which drafts up to 48 tokens and verifies them with its prompt kernels (235 drafted, 127 kept); on the hybrid 9B the reference gains 4 and 6 percent where lookup gains 32 and 59 in its two rounds, the reference's recurrent rollback costing it most of the drafts' worth.
- **Gates:** on the rig (9f5b5517, the draft-max default since changed): CTest 40 of 40 on an MI50 with the new `spec`; the suite on the CPU and on the MI50 passes every component but `raw-blocks`, which needs numpy the container lacks, and `mxfp4` on the MI50, skipped there; with drafts off, Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the MI50; with lookup at 4 and 16 drafts against off, the same ids, greedy and seeded, on a repeating and a plain prompt, from Qwen3-0.6B and Qwen3.5-0.8B Q8_0 on the CPU and the MI50, Qwen3-8B Q8_0, Qwen3-30B-A3B Q4_K_M and Qwen3.5-9B Q4_K_M on the MI50, and the 30B-A3B and the 9B on the CPU; a chat of two turns gives the same text with lookup on the CPU and the MI50; timing with drafts off on one MI50, pp512/tg128 alternating main and the change, Qwen3-0.6B 8484 and 8448 against 8456 and 8451 tok/s, tg128 389.7 and 380.2 against 379.9 and 380.1, Qwen3.5-0.8B 7334 and 7270 against 7333 and 7339, tg128 335.6 and 336.7 against 336.6 and 334.3, level. On the Radeon VII under Windows: CTest 41 of 41, the `f32`, `moe`, `qwen35`, `split` and `cli` components on the device, and Qwen3-0.6B Q8_0 and Q4_K_M giving the same ids with lookup at 4 and 16 drafts as without, greedy and seeded.
- **Merged** at `57a8c04a` with the fix to the qwen35 pause check's client timing (run 36917093047 green); drafting in the server and the price of a draft come with step 5.
- **Follow-up, move only:** this step adds about 180 lines to `src/model/runtime.hpp` (`Model::mark`, `Model::save`, `Model::rerun`, `Model::restore_mark`, `Model::drop_mark` and the mark's part of `Model::retract` and `Model::rewind`). The runtime's size rule asks features to add hooks and to move history operations out by concern, so a change that moves nothing else gathers the sequence-history operations (fork, checkpoint, keep, mark, retract, rewind and the rerun) into one file of `src/model/`, the owner section 1 of SPECULATIVE names, before step 4 adds the MTP layer's part of them. Done on `refactor/model-history`, below.
- **Gotchas:** text is delivered as a verify's rows are sampled, after its pass, so `generate`'s emit callback no longer runs before every model step when drafting; the proposer interface has `draft` alone, and `settle`, `block` and `reads` come with their first users (step 4's MTP); a mark takes only a free slot, so a request without one decodes that round with a single step.

## A restarted server keeps its checkpoints and its split (2026-10-01, branch fix/checkpoint-fit-settle, lands by fast-forward)

- **Bug:** the automatic checkpoint count was chosen before the fit read the cards' free memory again, so a server started while the card still held an ended server's memory (as production restarts it) fitted none and then, once the memory had come back, took the whole budget for KV: on one MI50, Qwen3.8-27B Q8_0 served with no `--ctx-size` started with 1 state checkpoint and 12288 KV tokens, then right after a killed server with 0 and 14720, then 20 s later with 1 and 12288 again; production's two MI50s showed 0 checkpoints over 262144 tokens, and a follow-up turn reused nothing.
- **A second case, the split:** production's two MI50s, restarted right after `docker stop` of the previous server, came up with the 27B's layers split unevenly, the process holding 22.1 GiB on the first card and 26.9 on the second where idle cards give 23.33 each, and decode at 18.6 tok/s against 19.1: over several devices the fit holds while one card still holds an ended server's memory, since the other takes its layers, so `settle` stopped at once.
- **Fix:** `fitted_kv` settles the budget without checkpoint slots first and chooses them after, and the startup line gives the KV tokens they took (`PlacedModel::checkpoint_kv_tokens`). Over several backends with a device that reports its free memory, where no shares are given or the budget is fitted, `settle` reads until that memory has risen no further for 20 reads, five seconds, or 120 reads have passed, and asks the fit of that reading alone (`level_first`), and the split places its layers by the reading the KV fit settled on; `serve` prints the split's plan as it starts.
- **Reproduced** on one MI50 and the CPU: Qwen3.8-27B Q8_0 served on the card, killed, and `bench --model` over `vulkan:0,cpu` started at once placed layers 0-2 on the card, which read 3.11 GiB free, and 3-63 on the CPU, where the card idle takes 0-45 at 31.97 GiB free; with the fix the start at once read 31.97 GiB and took 0-45 as idle. Qwen3-8B Q8_0, which the card holds whole either way, read 19.68 GiB free at once on main and 31.97 with the fix.
- **Tests:** `arch-qwen35`, a device whose free memory reads one byte for its first two reads: on the unfixed code it took 0 checkpoint slots beside 512 KV tokens, and with the fix the four asked for and the whole budget; two cards fitted to their free memory, the first's coming back after eight reads: on the head before the split's fix the second card took layers 0-3 and the first none, where idle cards split 0-1 and 2-3, and with the fix the plan, the budget and the checkpoints are the idle cards'; the KV tokens the checkpoints took are asserted in every fit case.
- **Gates:** on the rig, the test commit fails as above; with the fix at 90cd062f CTest 39 of 39 on an MI50, the suite on the CPU passes every component but `raw-blocks`, which needs numpy the container lacks, `cli`, `f32`, `moe`, `qwen35`, `split` and `server` pass on the MI50, Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the MI50, and `serve` over `vulkan:0,cpu` prints the split's plan before its startup line; rebased onto main d3817a38 without a conflict in code, CTest 39 of 39 again. At the first fix (cb032ec8): timing on one MI50, `bench --model` pp512/tg128 alternating main and the fix, Qwen3-0.6B 8503 and 8489 against 8492 and 8489 tok/s, tg128 393.7 and 388.5 against 393.1 and 392.0, Qwen3.5-0.8B 7264 and 7367 against 7235 and 7216, tg128 346.6 and 346.4 against 346.5 and 346.5, level, a single device taking no new wait; the 27B started three times, the second right after a killed server, kept 1 checkpoint taking 2432 KV tokens and 12288 KV tokens every time. XDEV reviewed the checkpoint half and asked for the KV cost to be asserted, which `arch-qwen35` does.
- **Cost:** a command that places a split by free memory, or a server fitting its budget over several devices one of which reports its memory, now waits at least five seconds, until that memory has risen no further for 20 reads, before it loads; `level_first` reads each device's memory once more than `budgets_for` does, which a later change may fold into the budgets read.
- **A void cell found on the way:** the Qwen3.5-0.8B identity cells of 8c's last gate runs (at f42414ca and 97b19ed4) read no file through a wrong mount, so both arms failed alike and compared equal; rerun with the file, 8c as landed gives the bytes of main before it in all 7 cells on the CPU and the MI50, as does this fix against its main.

## Prefix reuse for hybrid models through state checkpoints (2026-10-01, branch feat/qwen35-checkpoints, step 2 of SPECULATIVE, the qwen35 plan's 8c, lands by fast-forward)

- **Done:** the code, the docs and the tests (`arch-qwen35` checkpoints, `server-resume` hybrid checkpoints, the `qwen35` component's follow-up turn and pauses with and without checkpoints).
- **Gates so far:** CTest 39 of 39 on an MI50; the CPU suite passes every component; on one MI50 the `qwen35` and `server` components pass and every other but `perf`'s floor, which fails for main too at that load; Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the MI50. Timing on one MI50, `bench --model` pp512/tg128 alternating main and the change: Qwen3-0.6B 8491 and 8493 against 8495 and 8482, 390 and 393 against 391 and 391; Qwen3.5-0.8B over eight runs 7206 to 7354 against 7226 to 7365, tg128 347.3 to 348.1 against 346.4 to 347.3.
- **The use:** Qwen3.8-27B Q8_0 on one MI50 with production's flags (`--max-seqs 8 --ctx-size 32768`), an 8-turn chat through `/v1/chat/completions` as a chat UI sends it, arms main, the change, the change, main, every reply the same bytes in every arm. Time to first token by turn, prompt tokens, main and the change: 463, 1.99 and 2.07 s; 888, 3.55 and 1.85; 1343, 5.46 and 2.16; 1752, 7.29 and 2.17; 2118, 8.89 and 1.90; 2496, 10.20 and 1.78; 2987, 12.37 and 2.52; 3447, 14.29 and 2.31 s, the last reusing 2944 tokens. The fitted default took 3 checkpoint slots of 149.6 MiB, which lowered the KV budget from 32768 to 26688 tokens.
- **Against the reference server**, Qwen3.8-27B Q8_0 on one MI50 at default clocks: llmx main (e1877469) and the change (9cba9e54, whose fit gives the 27B the same 3 checkpoint slots and 26688 KV tokens as the head) with `--max-seqs 8 --ctx-size 32768`, and mx-llama.cpp's llama-server (build 10951, HIP) with 8 slots sharing a 32768-token cache and flash attention, at its defaults (32 checkpoints a slot, at least 8192 tokens apart), with `--ctx-checkpoints 64`, and with that and `--checkpoint-min-step 0`. Arms change, reference, 64, 64 with no spacing, main, main, 64 with no spacing, 64, reference, change, each a fresh server running two 14-turn chats through `/v1/chat/completions` as a chat UI sends them, streamed and greedy, every assistant turn sent back as its content without reasoning: one reasoning on every turn (replies of 240 tokens, some all reasoning) to 6227 prompt tokens, one with `enable_thinking` false (replies of 63 to 109 tokens) to 7461; then the user message of turn 6 edited and the conversation regenerated from there. Load average 5 to 27, the monitor's top processes the servers and the clients; the two runs of each server within 0.7 s of each other on every turn but the references' first.

  | | main | change | reference | reference, 64 | reference, 64, no spacing |
  |---|---:|---:|---:|---:|---:|
  | reasoning: follow-up TTFT p50 (turns 1 to 13) | 14.97 s | 2.32 s | 1.90 s | 1.96 s | 1.86 s |
  | reasoning: tokens read a follow-up turn, p50 | 3562 | 473 | 452 | 452 | 452 |
  | reasoning: turn 13 (6227 or 6317 tokens) | 27.50 s, 6227 read | 2.77 s, 531 read | 1.95 s, 490 read | 1.96 s, 490 read | 1.91 s, 490 read |
  | reasoning: edited turn 6 | 13.05 s, 3077 of 3077 read | 12.96 s, 3077 of 3077 | 9.92 s, 3122 of 3174 | 9.93 s, 3122 of 3174 | 4.68 s, 386 of 3174 |
  | no reasoning: follow-up TTFT p50 | 16.21 s | 2.77 s | 1.79 s | 1.81 s | 1.75 s |
  | no reasoning: tokens read a follow-up turn, p50 | 3831 | 585 | 436 | 436 | 436 |
  | no reasoning: turn 13 (7461 or 7470 tokens) | 33.07 s, 7461 read | 3.65 s, 677 read | 2.02 s, 543 read | 2.04 s, 543 read | 1.95 s, 543 read |
  | no reasoning: edited turn 6 | 14.24 s, 3368 of 3368 | 14.27 s, 3368 of 3368 | 10.72 s, 3355 of 3369 | 10.78 s, 3355 of 3369 | 6.87 s, 387 of 3369 |

  The change takes a follow-up turn from main's whole-history read to about the reference's, 6.5 and 5.9 times sooner at the median and 10 and 9 times at the last turn, and stays behind the reference by 0.4 and 1.0 s a turn. Two causes, each measured: llmx reads 400 to 500 new tokens more slowly than the reference (a fresh 463-token prompt 2.38 s against 1.69 to 1.84 s, a 433-token turn at depth 1.94 s against 446 tokens in 1.69 s), and without reasoning it reads about 140 tokens more a turn, the previous reply, whose decode rows a prompt does not fork (SPECULATIVE, decision 4), plus up to a block of the checkpoint's rounding. An edited earlier message loses the one checkpoint a request keeps: llmx reads the whole conversation again, as the reference at its defaults nearly does, while the reference with no spacing between its checkpoints reads only the edited message (4.68 and 6.87 s against 12.96 and 14.27). Raising `--ctx-checkpoints` alone changes nothing at this length.
- **Gates after the review's fixes**, on the code that lands: CTest 39 of 39 on an MI50 and 40 of 40 on the Radeon VII under Windows; on the CPU `dead-code`, `docs`, `cli`, `qwen35`, `split` and `server` pass, the last with the Qwen3-0.6B Q8_0 fixture and nothing skipped; `qwen35` and `server` pass on an MI50, `server` skipping only MXFP4, and on the Radeon VII at the review's first fixes; Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the MI50. XDEV rechecked every fix.
- **Left, from the comparison:** the reference reads a turn's 400 to 500 new tokens faster on the MI50, and a follow-up without reasoning reads the previous reply again (SPECULATIVE, decision 4); an edited earlier message would need more than one checkpoint a request.
- **Follow-up, proposed and not to be built now:** while the server is idle, read a finished request's reply again as prompt rows, in the prompt's row class, and keep a checkpoint after them, so a follow-up turn without reasoning forks past the reply and reads only its new message (SPECULATIVE, decision 4); the coordinator takes it to XDEV as a design amendment.
- **Gotchas:** a checkpoint position is whole blocks in every storage, so up to a block (64 tokens on Vulkan, 128 on the CPU) and the generation prompt are read again; a request whose prompt is shorter than a block keeps none; on a device a follow-up turn forks only once both prompts pass 449 tokens, where they take one tile split.

## A first admission forks only rows of its own classes (2026-10-01, branch fix/server-row-class, step 1 of SPECULATIVE, lands by fast-forward)

- **Done:** the failing test (`server-resume`: a follow-up turn on the CPU and on a device against its prompt on a fresh model; on main's code it fails at token 0, logprob -0.250535995 against -0.250535876), the change, and `tests/row_classes.hpp`, which `backend-group` and `backend-vulkan` run: every pair of 54 extents of one class, each side of every crossover and tile split, gives the same bits through a matmul, the routed products and attention (39962 pairs on the CPU, 2233 on an MI50).
- **Gates:** CTest 39 of 39 on an MI50 and 40 of 40 on the Radeon VII under Windows; the CPU suite passes every component; on one MI50 every component but `perf`, whose synthetic floor fails for main and the change alike at a load average near 25 (main 355 then 1680 tok/s, the change 532 and 541); Qwen3-0.6B and Qwen3.5-0.8B Q8_0 give main's bytes in all 14 identity cells on the CPU and the MI50; on the Radeon VII the `f32`, `qwen35` and `server` components pass. Timing on Qwen3-0.6B Q8_0 on one MI50, `bench --model` pp512/tg128, two rounds alternating main and the change at load averages of 17 to 25 shared with other work on the same cores: pp512 8458 to 8479 against 8470 to 8484 tok/s, tg128 343 to 392 against 368 to 390, level. XDEV reviewed it; its five points and the coordinator's three are in.
- **Gotchas:** on a device, prompts under 449 tokens of different lengths take different tile splits, so they share no rows even where a small model's bits agree, and a follow-up turn computes the previous reply again as prompt rows, about 1.3 s per 1000 reply tokens on the 8B on one MI50 (SPECULATIVE, decision 4).

