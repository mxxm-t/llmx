# cleanup/cpu-kernels gate (2026-09-26)

Main `27130f7` against the branch, rebased onto it, on the CPU of both machines and on one MI50.
The Linux runs used the branch at `79fc5e8`; the committed tree differs from it only in three comment lines of `cpu_backend.hpp` (one note moved to `docs/src/backends-cpu.md`) and in docs, and nothing in `src/` reads `__LINE__`, so the code is the same.
The Windows runs built the committed tree but for one of those comment lines.
After these runs the branch was rebased onto main `04e85b3`, whose commits change no file of the branch's code, and checked again against it on Linux (CPU builds, the harness, CTest and 15 CLI outputs): `linux-rebased.log`.

## Linux: the MI50 machine's CPU (EPYC 7262, 16 threads), no GPU device

- `linux-gate.sh`, run in a container with 6 CPUs, main mounted at `/cb` and the branch at `/cf`: both CPU builds, the kernel harness, the build-time requirement, CTest, CLI identity and the Python suite. Log: `linux-gate.log`, with the suite's in `linux-suite.log`.
- The script's `/t/ck_ident.cpp` is `kernel-harness.cpp` before the `sed` that adds 16 threads, and its `/t/srv_inner.py` is `server-threads.py`, which runs the suite's server component with its servers at `--threads 6`, the container's quota, since the automatic count oversubscribes it and the uncapped check then passes its limit. Log: `linux-server.log`.
- `linux-timing.sh`, run after the gate in a container with 6 CPUs: single-repeat `bench` runs, pairs alternating main, branch then branch, main, started without waiting on the load, which is recorded beside every run. Log: `linux-timing.log`.
- Other developers' jobs kept the load average at 16 to 25 on the host's 16 threads during the gate and at 13 to 53 during the timing.

## Linux: one MI50 through Vulkan

- After the timing, Vulkan-enabled builds of both trees (`-DLLMX_HAS_BACKEND_VULKAN=ON`), their generated SPIR-V compared, then on one free MI50 (rocm-smi index 3), 6 CPUs to the container: CTest and the suite with `--no-perf-floor --require-baseline --device vulkan:0` on the branch, and `generate -n 64 --temp 0`, `logits --file <1500 bytes of wiki.test.raw> --top 20` and `perplexity --ctx-size 128 --chunks 4` with `--device vulkan:0` on both, stdout compared. Log: `linux-device.log`, with the suite's in `linux-device-suite.log`.
- Eight bench pairs on the same card, `bench --device vulkan:0 --r 1`, alternating as above: `linux-device-bench.log`.

## Windows: Ryzen 7 5800X (8 cores, 16 threads), MSVC 14.50

- CPU builds of main and the branch (`cmake --build build-cpu --config Release`); CTest on the branch, 23 of 23.
- `kernel-harness.cpp` built with `cl /O2 /arch:AVX2` against each tree: `windows-harness.log`.
- `windows-identity.sh`: CLI identity at 16, 12 and 1 threads, `windows-identity.log`.
- `windows-timing.sh`: single-repeat `bench` runs in alternating pairs, `windows-timing.log`; the desktop's own programs were running.
- The Python suite on the branch's CPU build, 18 of 18: `windows-suite.log`.

## The Q6_K dot's contraction

`q6k-codegen.cpp` emits `dot_row_q6_K` alone.
Counting scalar FMA-family instructions (`vf[n]madd*ss`, `vf[n]msub*ss`) in it at `-O3 -mavx2 -mfma -mf16c`:

| Build | main | branch | branch with main's plain expression |
|---|---:|---:|---:|
| clang 18.1.3 | 42 | 0 | 8 |
| GCC 13.3 | 0 | 0 | 8 |
| clang 18.1.3, `-ffp-contract=fast` | | 8 | |
| GCC 13.3, `-ffp-contract=fast` | | 0 | |

Eight of main's 42 under Clang are the AVX2 path's `acc += d * (float)sc * s`, each a `vfmadd213ss` after the group's horizontal sum; the rest are in its scalar branch, which the branch removes.
The last column puts main's `acc += d * (float)sc * s` back in place of the two intrinsics: with the scalar branch gone, GCC fuses it as Clang does, which is why the branch rounds the product and the sum separately.

Clang 18 in WSL also built the whole CPU tree (`-DCMAKE_CXX_COMPILER=clang++`, Release) without a warning.
