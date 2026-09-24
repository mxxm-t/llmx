# Backend architecture and failure-path audit (2026-09-25)

Reviewed published commit `e475c3f58c31944f397d96abd7b079fc90f5f082`.
This is a source audit with focused failure probes, not a new numerical or
performance gate. LDEV's attention changes and the activation-range candidate
are separate. No runtime or test-suite source is changed by this checkpoint.

## Architecture and development rules

The backend interface uses device-neutral buffers, slices, tickets and KV
views. Physical KV layout and worker/queue policy stay inside the backends;
placement and logical request state stay in model/server code. One submitting
thread per backend is the stated contract; a shared backend is not a concurrent
public API. CPU eager execution and Vulkan queued execution implement that
contract without a second scheduling framework. Future vendor backends need
real implementations of the whole interface, not placeholder classes.

A mechanical scan of all 33 files in `src/backends` found zero includes of
model/inference/server/CLI/Hub headers, zero runtime environment reads, and
zero non-ASCII files. This does not establish mathematical correctness of every
kernel. The reviewed code has no new external library dependency; Vulkan's
SDK/loader boundary is documented. The existing quant-to-format type-ID
coupling remains the architecture's disclosed exception.

CPU pool failures drain participants before throwing, and KV growth publishes
storage only after allocation. No new ownership failure was found on validated
model calls. The existing activation reciprocal-range bug has a separate fix
at `91d2ba3`; its outstanding depth/performance gates remain outstanding.

## Findings

| Priority | Finding and trigger | Observed | Required |
|---|---|---|---|
| P2 | Vulkan kernel construction fails, then the same kernel is retried | One shader module, descriptor layout and pipeline layout orphaned after teardown | No orphaned handles |
| P2 | A diagnostics backend with a query pool is destroyed | Zero query-pool destroy calls | One destroy call after retirement |
| P2 | Clearing two cached padded weight ranges fails on the second retirement-vector allocation | Two cache entries remain, one with a null copy | Cache remains safe to reuse or invalidates atomically |
| P2, direct API boundary | CPU receives two row runs ending at 3 then 2 for a two-row batch | Callback runs three rows before rejection; equal-kind runs are silently collapsed | Validate all metadata before executing rows |

- `vulkan_backend.cpp:2219-2282`: `kernel()` writes resource handles directly
  into the cached `Kernel` before construction succeeds. Retrying overwrites
  the old handles. Construct with cleanup on failure and publish a complete
  kernel only after success, or reliably reset the partial entry.
- `vulkan_backend.cpp:892-907,2317-2323`: dispatch creates `queries_`, but the
  backend destructor never calls `vkDestroyQueryPool`. The probe installs a
  fake populated handle and counts destruction; source inspection connects
  this to the real timestamp creation path. It does not claim a real-driver
  memory-leak measurement.
- `vulkan_backend.cpp:1906-1910`: `drop_padded()` moves cached pointers one by
  one. Allocation failure after the first successful move leaves that entry
  null because the final map clear is never reached. A subsequent matching
  `padded_f32()` lookup dereferences the null copy at line 1903. The probe
  inspects the state instead of deliberately crashing. Reserve retention
  capacity before moving entries, or use an equally small transactional fix.
- `cpu_backend.hpp:344-359`: `each_run()` checks the final endpoint and the
  endpoints of merged groups, but does not validate all runs before invoking
  callbacks. With `{3,1},{2,2}` and `nbatch=2`, the first callback receives
  three rows; `{3,1},{2,1}` is silently coalesced. This requires malformed
  direct backend metadata; no model-generated invalid run was demonstrated.
  Grouped dispatch must use the same validation contract. The probe counts
  callbacks and deliberately performs no out-of-bounds memory access.

These failures are distinct from the previous allocation-lifetime checkpoint's
constructor, KV growth, padded-copy replacement and arena-overflow cases.
They do not invalidate those specific passing results, but show their scope
was narrower than general backend exception safety. Add focused failure and
retry regressions rather than loosening numerical gates or repeating only
happy-path model tests.

The unused Vulkan `todo()` migration helper can be removed as cleanup. It has
no callers and is not evidence of an unimplemented runtime operation. Generic
operand instrumentation, a new framework, and a backend rewrite are not
recommended by this review.

## Reproduction and scope

`probe.cpp` includes the unchanged Vulkan implementation using the existing
private test friend. It opens the local Radeon VII, substitutes kernel handle
creation/destruction and a query handle, and injects one C++ allocation failure.
No fake handle is submitted for GPU execution. The three findings reproduce;
`probe.log` contains the counts. `cpu-probe.cpp` exercises the actual CPU row
iterator with callbacks that count rows; both malformed-input cases reproduce.
The build scripts retain exact MSVC commands and original temporary paths;
change those paths to reproduce elsewhere. Generated Vulkan shader includes
come from the reviewed main build. These are diagnostic probes, not permanent
CTest additions. Logs retain their original byte encoding.

`probe.json` records binary/source/log hashes for the Vulkan probe.
`source-scan.json` records all reviewed backend source hashes.
`artifacts.json` records the saved review artifacts. Independent source review
confirmed the three Vulkan probes and the CPU finding. No performance was
measured while the long-context correctness workload ran. No new HF, Linux,
MI50, shader-arithmetic or other-vendor validation is claimed here.
