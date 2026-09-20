# backends/cpu/prefill_placement.hpp

Backend-private Windows CPU placement for one synchronous prefill call.
`CpuBackend::run_prefill` owns the helper and dispatches apply/restore on the
same persistent pool participants. Placement-specific Windows types stay out of the model and CLI layers.

Only six workers on a single processor group with homogeneous efficiency
classes and at least six allowed physical cores qualify. The first allowed
logical processor of each of the first six cores is selected deterministically.
Placement stays within incoming process and thread masks; cleanup restores the original masks.
Other thread counts, unsupported topology and non-Windows builds pass through.
There is no runtime flag, environment setting or NUMA memory policy.

Apply is all-or-fallback: a partial or unverified setup is restored before the
body runs without placement. Original masks and thread identities are checked
on cleanup, including when the body throws. Cleanup retries participants still needing restoration once and reports the
first cleanup failure even if recovery succeeds. Persistent OS refusal cannot
guarantee restoration or safe reuse; it never becomes a silent success.

The helper stores policy and per-participant state, not model state. It adds no
concurrent-submission support. Each backend still requires a single caller.
