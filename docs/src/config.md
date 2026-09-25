# `src/config.hpp` - build configuration

Single place for compile-time build knobs.

- CMake regenerates it via `configure_file` into the build dir and adds that dir
  to the include path; the checked-in copy is what the plain `build.bat` path
  uses. Keep the two in sync when adding build knobs.
- **Backends** are the only things that need compile-time config, because GPU
  backends pull in heavyweight SDKs (ROCm/CUDA/SYCL/Vulkan). CPU is always on (no
  external deps). `LLMX_HAS_BACKEND_VULKAN` builds the Vulkan backend; the
  ROCm, CUDA and SYCL names reserve future gates and build no code.
- **Model architectures** are *not* here: the model reads
  `general.architecture` at run time; dense Qwen3 and `qwen3moe` are
  implemented.
- **Split mode / node count** are *not* here: layer splitting is selected at
  runtime through `--device` and `--layer-shares`; tensor groups and node
  execution remain planned.

Defines:

- `LLMX_RELEASE_VERSION` identifies the release.
- `LLMX_BUILD_REVISION` is generated at build time: `g<commit>[.dirty]`, or
  `unknown` without usable Git metadata. `LLMX_VERSION_STRING` combines both.
  CMake refreshes metadata every build and rewrites the header only when it
  changes. Plain `build.bat` writes the same header before compiling.
  Direct compiler builds without that header retain the `unknown` fallback.
  Plain builds stop before compilation if writing the version header fails.
- Update the CMake project version and the fallback config's version string
  together for a release. Rebuilding does not bump or tag a release.
- `LLMX_HAS_BACKEND_VULKAN`: whether the Vulkan backend is compiled in. The
  CPU backend always is, and another backend gets its flag when it exists.
  Thread counts are runtime flags, not build options.
