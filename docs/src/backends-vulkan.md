# `src/backends/vulkan/` - the Vulkan backend

The `Backend` of `backend.hpp` over a Vulkan 1.2 compute queue, in one
translation unit (`vulkan_backend.cpp`, built only with
`LLMX_HAS_BACKEND_VULKAN=ON`) and the GLSL kernels under `shaders/`, which
`glslc` compiles at build time into arrays the unit includes. The loader is
opened at run time, so a build carries no link dependency; the design,
kernel notes and measurements are `docs/VULKAN.md`.

- `make_vulkan_backend(index, diagnostics)`, `vulkan_device_name`: open
  the loader, pick the device, require what the kernels need (Vulkan 1.2,
  subgroups of 32 lanes or more, 16-bit integers, timeline semaphores,
  push descriptors); anything missing throws `VulkanUnavailable`, which
  the test skips on and the CLI reports, as does a loader with no driver
  behind it.
- `vulkan_kernel_statistics` returns the driver's per-kernel registers,
  shared memory and scratch when the device serves them, which the test
  prints after its checks. With `diagnostics` the backend also captures
  the driver's disassembly of each kernel, which
  `vulkan_kernel_representations` returns and `backend-vulkan --isa DIR`
  writes one file per kernel. Both are read-only reporting: nothing in
  the runtime path depends on them.
- Buffers are `VulkanBuffer`, device-local or host-visible; every op is
  recorded into a ring of command buffers and submitted in chunks of 64
  dispatches, so the device starts a pass while the host records the rest.
  Small per-call inputs go through a host-visible arena per ring slot; a
  scratch outgrown mid-pass retires with the slot rather than being freed
  while recorded commands still name it.
- `matmul` and `matmul_group`: narrow batches take the row kernel, one
  module per family of types, reading quantized rows against 16-bit
  integer activations (`shaders/xquant.glsl`) that the producing kernel,
  the norm, the SiLU or the attention, writes beside its output and tags;
  wide batches take the tile kernel with float activations. Which of the
  two, and what shape the tile has, comes from
  `backends/device_profile.hpp`: the row count where the tile starts
  winning moves with how much work a row carries, and the tile's height
  is a specialization constant the backend picks from the device's
  compute units so a small call still fills the card.
- `kernel_statistics` and `kernel_representations` report what the driver
  made of each kernel, registers and occupancy from the first and its
  disassembly from the second, the latter only for a backend opened with
  `diagnostics`. Neither is on the runtime path; `backend-vulkan` prints
  the statistics and `--isa DIR` writes the disassembly.
- The KV cache is `VulkanKVStorage`, blocks of 64 tokens in f32 or f16,
  written and read through a view table so every cache kernel runs once
  per layer over every view of a batch; attention splits a batch between
  the tiled kernel and the per-row kernel with its history splits and
  merge.
- `kv_variant` picks the shader module for a storage's K and V types.
