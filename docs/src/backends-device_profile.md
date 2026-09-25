# `src/backends/device_profile.hpp` - what a GPU backend shapes its kernels by

Device numbers in namespace `backend`, apart from any one vendor's API. The
Vulkan backend fills and reads them today; a second GPU backend would share
them (`docs/ROADMAP.md` 4b). Branch on what a device reports, never on its
vendor name.

- `DeviceCaps`: what the device reports, filled by the backend from its own
  API: subgroup width, compute units, the names the device and its driver
  give themselves, and whether it has an integer dot product. A capability
  decides which kernel to launch.
- `DeviceProfile`: numbers found by measuring the kernels, which nothing in `DeviceCaps` implies: the lanes a row takes at most in the integer-dot row kernels, the batch rows and prompt extents from which a matmul, a routed projection or attention takes a tile kernel, the tile split and fill targets, attention's history split sizes, dispatches per submission, and whether the matmuls take the integer dot.
  A profile number shapes the same kernel.
  A number a shader's layout fixes, such as the lanes that share a quant block or the query rows of an attention tile, is a constant beside the kernels instead, since another value would break the kernel rather than tune it.
- `TunedDevice`, `tuned_devices()`: the devices and drivers the profile was tuned on, keyed by a substring of each name, each row with a function that assigns the numbers measured there by name.
  It is the table to extend when bringing up hardware, with the sweeps in `docs/VULKAN.md`.
- `profile_for(caps)`: the defaults, tuned by the first row whose device and driver both match; a row that prefers the integer dot applies only where the device has it.
- `tile_from_for`, `moe_tile_from_for`, `tile_rows_for`: the batch rows from
  which a matmul of these types and width takes the tile, the prompt extent
  from which a routed projection of this weight type does, and a tile's
  height from the call's shape and the device's compute units.
