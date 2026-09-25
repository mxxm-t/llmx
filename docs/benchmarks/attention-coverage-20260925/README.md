# Vulkan attention coverage checkpoint (2026-09-25)

Base: published main `c602ace35a06597d832b9d559458d39fd4a872ad`.
This change extends native tests and corrects their documentation. No runtime
source, shader, build configuration, numerical bound or CLI flag changes.

The existing CPU oracle now covers 80 combinations: head widths 32, 40, 64,
128 and 256; query/KV head ratios 1, 2, 4 and 8; and all four F32/F16 K/V
pairs. Width 40 takes the non-vector kernel; the other widths take the vector
kernel on the tested device. Histories of 2100 and 3000 exercise grouped heads.
A 40-token history beside a 3000-token history checks both output rows against
separate calls, bit for bit. All outputs must be finite. Repeated attention
cases under unrelated matmul widths were removed; distinct old cases remain.

The host-visible buffer check also now uses its 777-byte fixture's size for
both the write and comparison, replacing two invalid 1000-byte requests.
The earlier passing run is retained as pre-correction evidence.

| Validation | Observed | Required |
|---|---:|---|
| Final native suite | 24/24 passed, zero skipped | All tests |
| Attention shape/cache combinations | 80 passed | All combinations |
| CPU/GPU attention error | All values within bound | `1e-4 * (1 + abs(cpu))` |
| Mixed-history output versus separate GPU calls | Both rows bit-identical | Exact, finite |

The final run used Windows, MSVC 19.50 Release and an AMD Radeon VII.
[Report](report.json), [commands](final/commands.json), [CTest summary](final/native.log)
and [complete native output](final/LastTest.log) retain the result and identities.
The drivers are preserved byte-for-byte; they were run from the worktree root,
so copy one there if replaying it. The command records are the direct
reproduction reference and use a fresh build directory.

No new HF or performance claim is made. Runtime source and build configuration
match the base; these checks supplement the external HF gate. The bandwidth
lines in native output are diagnostic, with evidence compression concurrent,
and are not controlled performance measurements. MI50 and other Vulkan devices
were not executed for this checkpoint.
