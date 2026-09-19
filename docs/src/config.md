# `src/config.hpp` - build configuration

Single place for compile-time build knobs.

- CMake regenerates it via `configure_file` into the build dir and adds that dir
  to the include path; the checked-in copy is what the plain `build.bat` path
  uses. Keep the two in sync when adding build knobs.
- **Backends** are the only things that need compile-time config, because GPU
  backends pull in heavyweight SDKs (ROCm/CUDA/SYCL/Vulkan). CPU is always on (no
  external deps). The `LLMX_HAS_BACKEND_*` GPU names currently reserve future
  gates; enabling an option does not build a vendor backend.
- **Model architectures** are *not* here: runtime metadata selection is planned;
  only dense Qwen3 is implemented today.
- **Split mode / node count** are *not* here: they are planned runtime parameters.

Defines:

- `LLMX_VERSION_MAJOR/MINOR/PATCH` and `LLMX_RELEASE_VERSION` identify the release.
- `LLMX_BUILD_REVISION` is generated at build time: `g<commit>[.dirty]`, or
  `unknown` without usable Git metadata. `LLMX_VERSION_STRING` combines both.
  CMake refreshes metadata every build and rewrites the header only when it
  changes. Plain `build.bat` writes the same header before compiling.
  Direct compiler builds without that header retain the `unknown` fallback.
- `LLMX_HAS_BACKEND_CPU/ROCM/CUDA/SYCL/VULKAN`
- `LLMX_DEFAULT_THREADS` is declared but not consumed by the CPU backend.
  Actual thread control uses CLI flags and backend hardware concurrency.
