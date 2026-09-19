# `src/config.hpp` — build configuration

Single place for compile-time build knobs.

- CMake regenerates it via `configure_file` into the build dir and adds that dir
  to the include path; the checked-in copy is what the plain `build.bat` path
  uses. Keep the two in sync when adding build knobs.
- **Backends** are the only things that need compile-time config, because GPU
  backends pull in heavyweight SDKs (ROCm/CUDA/SYCL/Vulkan). CPU is always on (no
  external deps). Each GPU backend is gated by `LLMX_HAS_BACKEND_*`.
- **Model architectures** are *not* here: all are compiled in and selected at
  runtime from the model file's metadata.
- **Split mode / node count** are *not* here: they are runtime parameters.

Defines:

- `LLMX_VERSION_MAJOR/MINOR/PATCH` and `LLMX_VERSION_STRING`
- `LLMX_HAS_BACKEND_CPU/ROCM/CUDA/SYCL/VULKAN`
- `LLMX_DEFAULT_THREADS` (0 = auto / hardware concurrency)
