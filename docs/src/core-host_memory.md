# `src/core/host_memory.hpp` - available host memory

`core::host_memory_available()` returns the bytes of physical memory the operating system says a process can still take without swapping: `GlobalMemoryStatusEx`'s available physical memory on Windows, `MemAvailable` from `/proc/meminfo` on Linux (free memory plus page cache the kernel can reclaim), falling back to available pages from `sysconf`, and 0 when none of these can be read.

The CPU backend reports it as `memory_available()`, which a placement over several devices is fitted against (`docs/MULTI-DEVICE.md`). Weights on the host read the mapped file in place, so they are not what this bounds; caches, activations and anything a loader materializes are. The figure is a moment's reading on a machine with other work, not a reservation.
