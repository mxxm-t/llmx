# `src/core/host_memory.hpp` - available host memory and the page size

`core::host_memory_available()` returns the bytes of physical memory the operating system says a process can still take without swapping: `GlobalMemoryStatusEx`'s available physical memory on Windows, `MemAvailable` from `/proc/meminfo` on Linux (free memory plus page cache the kernel can reclaim), falling back to available pages from `sysconf`, and nothing (`std::nullopt`) when none of these can be read.

The CPU backend reports it as `memory_available()`, which a placement over several devices is fitted against (`docs/MULTI-DEVICE.md`); `infer::place_model` passes it to the fit as the host's free memory, and `infer::load_model` leaves a payload larger than it unread rather than reading its pages twice ([load](inference-load.md)). Weights on the host read the mapped file in place, so they are not what this bounds; caches, activations and anything a loader materializes are. The figure is a moment's reading on a machine with other work, not a reservation.

`core::page_size()` is the size of a page of memory (`GetSystemInfo`'s `dwPageSize` on Windows, `sysconf(_SC_PAGESIZE)` elsewhere), read once; it throws when the operating system gives no positive size, since every use steps or multiplies by it. `format::MappedFile::drop` releases whole pages of it, and `gguf::warm` reads one byte of every page.
