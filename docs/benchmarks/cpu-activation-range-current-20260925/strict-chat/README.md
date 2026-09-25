# CPU activation native-chat depth gate (2026-09-25)

The separately frozen native-chat comparison completed all nine calls and
passed an independent audit of the raw response files. Each model ran on
fresh baseline, repeat and candidate servers. All three responses agreed
exactly in token IDs, text, finish reason and counts, with zero reused tokens.

| Model | Prompt tokens | Generated tokens | Calls | Finish |
|---|---:|---:|---:|---|
| Qwen3-0.6B Q8_0 | 19,820 | 244 | 3/3 | EOS |
| Qwen3-0.6B Q4_0 | 19,820 | 302 | 3/3 | EOS |
| Qwen3-0.6B Q5_K_M | 19,820 | 229 | 3/3 | EOS |

`plan.json` freezes the request, model paths and identities, old executable
identities, call order and flags. The request uses native `/v1/chat`, the
embedded template and default thinking, temperature zero, seed zero and no
generation cap. Each CPU server uses six threads, ubatch 128, F16 K/V and a
32,768-token context. No manual stop token or shortened prompt was introduced.
The plan SHA-256 is
`7b269b13a140398e97efbab605db555dc9f74177ac1e50aee8cded143c78c170`.

`runs/` preserves all response, command and server-log files. `results.json`
and `complete.json` record nine successful calls; the driver exited zero.
The per-server exit code of one follows deliberate Windows termination after
the successful request. It is not an inference failure. No driver or server
was left running after completion.

`audit/audit_completed.py --final` checks the exact planned order, raw response
hashes, command flags, nonempty EOS, counts, zero reuse and within-model
response equality without trusting the runner's `matches` booleans. Its
retained final report is `audit/final-9-20260925T015124456253Z.json`.
Saved HF/Jinja prompt evidence agrees on all 19,820 input IDs. This stdlib
audit checks that evidence's consistency; it does not rerun HF tokenization.

This is output self-consistency at depth, not independent HF numerical or
semantic correctness. The separate real-model HF NLL and synthetic numerical
gates remain necessary. It also does not complete the original raw-completion
EOS experiment: that Q8 baseline filled its 32,768-token context and ended by
length, without a repeat or candidate. That historical outcome remains
retained in the [original raw-completion evidence](../../cpu-activation-range-perf-20260924/raw-completion/README.md).

The servers used frozen old baseline `5c8e89c346a5` and candidate
`842f453ed22c`. `audit/source-scope-bridge.json` and its full source/test diff
connect the candidate to current integration `356b7457398a`: CPU backend,
backend interface and model changes are comments; CLI executable changes
are in perplexity/benchmark paths, and Vulkan ownership changes are outside
this CPU run. This is a source-scope comparison, not binary identity or
performance equivalence. Current integration has separate native/HF checks
and a separately frozen final performance matrix.

Durations are diagnostic only. Earlier correctness/build activity overlapped
parts of this gate. They must not be compared as performance measurements.
`artifacts.json` lists copied files and raw-byte hashes. Binaries and object
files are excluded; the build helper and source retain reproduction context.
