# `src/backends/devices.hpp` - backends named by a device spec

Where a device spec becomes a backend, for every caller: the CLI's `--device`, the tools, and the tests.

- `canonical_device(spec)`: a spec in its one spelling, `cpu` or `vulkan:N`; `vulkan` alone and `vulkan:00` are `vulkan:0`. Another backend name or a malformed index throws.
- `device_specs(value)`: a comma-separated list of specs, each canonical, in order. An empty entry or a device listed twice throws, since two stages would drive one backend and its free memory would count twice.
- `make_backend(spec, diagnostics)`: the backend a spec names; `diagnostics` asks a device backend to time its kernels (`bench --profile`). A Vulkan spec in a build without the Vulkan backend throws.
- `make_backends(specs, diagnostics)`: one backend per spec, in the list's order.

How the model is placed over those backends is the model layer's (`infer::place_model`, `model/arch_qwen.hpp`).
