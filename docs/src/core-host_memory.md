# `src/core/host_memory.hpp` - the host memory a process can still take, the page size and owned pages

`core::host_memory_available()` returns the bytes of memory this process can still take without swapping or passing a memory limit, read now: the fewer of the host's available memory and the room the process's memory limits leave, each only when it could be read, and nothing (`std::nullopt`) when neither could (`host_memory_available(host, room)` combines the two).
A container run with `--memory 8g` on a host with 26 GiB available is given what is left of its 8 GiB; outside a limit it is the host's figure.

- The host's available memory, `system_memory_available()`: `GlobalMemoryStatusEx`'s available physical memory on Windows, `MemAvailable` from `/proc/meminfo` on Linux (free memory plus page cache the kernel can reclaim), falling back to available pages from `sysconf`.
- The room the memory limits leave, `memory_limit_room()`: the cgroup memory limits on Linux (`cgroup_memory_room`) and the job object's memory limits on Windows (`job_memory_room`), below; nothing without a limit or when it cannot be read.

The CPU backend reports it as `memory_available()`, which a placement over several devices is fitted against (`docs/MULTI-DEVICE.md`); `infer::place_model` passes it to the fit as the host's free memory, and `infer::load_model` leaves a payload larger than it unread rather than reading its pages twice, reads around the file cache in `auto` when the bytes it streams are more than it, and refuses a direct load whose host reads more weights in place than it ([load](inference-load.md)). Weights on the host read the mapped file in place, so they are not what this bounds, except in a direct load, where they are the loader's own copy; caches, activations and anything else a loader materializes are. The figure is a moment's reading on a machine with other work, not a reservation.

## The cgroup memory limits (Linux)

`cgroup_memory_room(proc_self_cgroup, mountinfo, read)` reads the limits in the process's own cgroup and in each above it up to its mount's point, which [cgroup](core-cgroup.md) finds from the text of `/proc/self/cgroup` and `/proc/self/mountinfo`, through `read`, which gives a file's text or nothing; `cgroup_memory_room()` passes the real files and reader, and returns nothing off Linux.
Each cgroup's room is its limit less its working set, 0 when the working set is past the limit, and the fewest over every cgroup read wins, since a limit on a parent holds every cgroup under it.
The working set is the usage less the inactive file pages the cgroup's own `memory.stat` gives, 0 when those are more than the usage.
A cgroup's usage counts the page cache charged to it, and the kernel reclaims its inactive file pages before it would kill the process, so they are available in the sense `MemAvailable` counts the host's reclaimable cache available: a model file read once in a container leaves the next load, or the loader's copy decision, what the limit leaves beside the process's own memory, where the limit less the usage would leave almost nothing.

- cgroup v2 (`cgroup_v2_memory_room`): `memory.max` less the working set (`memory.current` less `inactive_file`), where "max" is no limit;
- cgroup v1 (`cgroup_v1_memory_room`), in the hierarchy mounted with the `memory` controller: `memory.limit_in_bytes` less the working set (`memory.usage_in_bytes` less `total_inactive_file`, which counts the cgroups under it as the usage does), where a limit of 2^62 bytes or more is no limit, since v1 writes none as the largest signed 64-bit value rounded down to a page (9223372036854771712 with 4 KiB pages).

A `memory.stat` that is missing, lacks the key or gives it as anything but one decimal number leaves the usage whole, the limit less the usage.
A limit or usage that is not one decimal number, a limit without its usage and a cgroup whose files are missing are passed over, so a limit that cannot be read changes nothing, and with none read the host's figure stands.
Active file pages count as used, since the kernel takes them back only once they turn inactive, and `memory.high`, which throttles rather than kills, and swap limits are not read.

## The job object's memory limits (Windows)

`job_memory_room()` asks `QueryInformationJobObject` for the extended limits of the process's own job; a job it is nested in is not read.
With the process memory limit flag (`job_limit_process_memory`, `JOB_OBJECT_LIMIT_PROCESS_MEMORY`) the room is the process memory limit less the process's commit charge (`GetProcessMemoryInfo`'s `PagefileUsage`); with the job memory limit flag (`job_limit_job_memory`, `JOB_OBJECT_LIMIT_JOB_MEMORY`) it is the job memory limit less the job's committed memory (`JobObjectLimitViolationInformation`'s `JobMemory`); with both, the fewer.
Both limits bound committed memory, not physical memory, so beside the host's available physical memory the fewer of the two is what a process can take; a limit whose commit cannot be read is passed over.
`job_memory_room(limit_flags, process_limit, process_commit, job_limit, job_commit)` reads the fields, 0 past a limit and nothing without either flag.

`tests/host_memory.cpp` (CTest `host-memory`) holds the file texts, the walk over a file system held in a map, the job's fields and the combination, and prints what this process can take, the host's figure and its limits' room.

## The page size and owned pages

`core::page_size()` is the size of a page of memory (`GetSystemInfo`'s `dwPageSize` on Windows, `sysconf(_SC_PAGESIZE)` elsewhere), read once; it throws when the operating system gives no positive size, since every use steps or multiplies by it. `format::MappedFile::drop` releases whole pages of it, and `gguf::warm` reads one byte of every page.

`core::HostPages(bytes)` is page-aligned memory the process owns, rounded up to whole pages (`VirtualAlloc` on Windows, an anonymous private `mmap` elsewhere) and given back when it goes; it moves and does not copy, and an allocation the operating system refuses throws. A size past the largest whole number of pages a `size_t` holds cannot be rounded up, and it and `reserved` refuse it with a `std::length_error` naming the size rather than wrap it to an empty object. The loader's read ring is up to four of them, fewer when a load has fewer pieces to read (`infer::detail::stream`), since a read aligned to the page, and to the file system's block beyond it, is what a file system serves whole and what a read around the file cache needs.

`HostPages::reserved(bytes)` is address space with no memory behind it (`VirtualAlloc(MEM_RESERVE)`, or an anonymous `PROT_NONE` mapping with `MAP_NORESERVE`); `commit(offset, bytes)` gives memory to the whole pages covering a range of it (`MEM_COMMIT`, or `mprotect` to read and write), harmlessly again over pages already committed, and `decommit(offset, bytes)` takes the memory of the whole pages inside a range back, keeping the address space (`MEM_DECOMMIT`, or `madvise(MADV_DONTNEED)` and `mprotect` to no access). A range outside the reservation is a `std::logic_error`. A direct load lays a copy of each file out this way for the weights the CPU reads in place, committing only the ranges it reads into and giving back after the reads the pages that hold none of those weights, so each weight keeps the address offset within its page that a mapping would give it.
`tests/file_reader.cpp` (CTest `file-reader`) checks owned pages and their moves, and a reservation's rounding, a size past the largest whole number of pages a `size_t` holds refused by name by both and the largest that rounds refused by the operating system, commits that start and end inside a page and repeat, commits outside it refused, and a decommit that keeps the pages it covers only in part.
