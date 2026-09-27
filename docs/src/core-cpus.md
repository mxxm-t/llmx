# `src/core/cpus.hpp` - the CPUs a process may use and the automatic worker count

`core::automatic_threads()` is the worker count a pool takes when it is given none, the one owner of that count: the CPU backend's constructor takes it ([cpu](backends-cpu.md)), so `--threads` omitted or 0 means it in every command, `serve` and `bench` included (`docs/USAGE.md`, Threads).
It is the fewest of three counts, each taken only when it could be read, 4 when none could, and never below 1 or above 64:

- the hardware threads, `std::thread::hardware_concurrency()`;
- the CPUs the process's affinity allows (`affinity_cpus`), below;
- the CPUs the process's CPU quota allows, rounded up (`quota_cpus`): its cgroup CPU quotas on Linux (`cgroup_cpus`), its job object's CPU rate hard cap on Windows (`job_cpus`), below.

A container started with `--cpus 6` on a 16-thread host is allowed 6 CPUs of time by its quota while its affinity still lists all 16, so the hardware threads and the affinity alone start 16 workers that the quota throttles together; the quota makes it 6.
The sources are read each time it is called, which is once per CPU backend, so a count changed on a running container reaches the next backend made.

## The affinity

On Linux `affinity_cpus` counts the mask `sched_getaffinity` gives.
The mask is allocated with `CPU_ALLOC` and doubled while the kernel refuses it as too small (`EINVAL`), which it does when it has more possible CPUs than the mask holds, up to 2^20 CPUs.

On Windows it goes by the processor groups the process's threads lie in, not the machine's.
`group_affinity()` gives the one group they all lie in, the process affinity mask in it and the group's active processors, from `GetProcessGroupAffinity` and `GetProcessAffinityMask`, and nothing when they span several groups or a mask cannot be read.
With one group the count is that mask's bits, so a restriction such as `start /affinity` or a job object's affinity holds, and a process on a machine of several groups, which Windows 10 keeps in its primary group, counts that group's processors rather than the machine's.
With several groups the process affinity mask is not given, so the count is every active processor of those groups.
`group_affinity()` is also where the CPU backend's prefill placement takes the process mask from ([placement](backends-cpu-placement.md)).

## The cgroup CPU quota (Linux)

`cgroup_cpus(proc_self_cgroup, mountinfo, read)` finds the process's cgroups in the text of `/proc/self/cgroup`, their mounts in the text of `/proc/self/mountinfo`, and reads their files through `read`, which gives a file's text or nothing; `cgroup_cpus()` passes the real files and reader, and returns nothing off Linux.
For cgroup v2 it takes the `0::/path` line (`cgroup_v2_path`) and the `cgroup2` mounts (`cgroup_v2_mounts`), and reads `cpu.max`; for v1 it takes the first line whose controllers list `cpu` (`cgroup_v1_cpu_path`) and the `cgroup` mounts whose super options list `cpu` (`cgroup_v1_cpu_mounts`), and reads `cpu.cfs_quota_us` and `cpu.cfs_period_us`.
A path, or a mount's root, that is not absolute or climbs with `..` is not read.

`/proc/self/cgroup` names the path in the cgroup hierarchy as the process's cgroup namespace sees it, which in a container without its own namespace is the path on the host, while a mount can hold only a part of the hierarchy: mountinfo's fourth field is the hierarchy's directory mounted there.
`cgroup_directory` takes the first mount whose root holds the process's path, and the path below that root is its directory under the mount point; a path no mount's root holds is not read.
So a v1 container's cgroup `/c` mounted at `/sys/fs/cgroup/cpu,cpuacct` is read at the mount point, a child cgroup `/c/inner` under it at `/sys/fs/cgroup/cpu,cpuacct/inner`, and a cgroup inside the container that happens to share a name with a directory on the host's path is never taken for the process's own.
From that directory it reads each cgroup up to the mount point, and the smallest limit found wins, since a quota on a parent holds every cgroup under it; cgroups above the mount's root are not visible there and are not read.
A hybrid machine, v1 controllers beside a v2 hierarchy, is read both ways, and the files that exist give the limit.

## The job object's CPU rate (Windows)

`job_cpus()` asks `QueryInformationJobObject` for the CPU rate control of the process's own job, the one a process-isolated Windows container's `--cpus` sets; a job it is nested in is not read.
`job_cpus(control_flags, rate, processors)` reads it: with the enable and hard cap flags set (`job_rate_enable`, `job_rate_hard_cap`, the values Windows gives them) a rate of `rate` cycles in each 10000 over `processors` active processors, those of every group, is that many CPUs rounded up, so 1250 over 16 processors is 2.
Without both flags, as under a weighted rate or a minimum and maximum rate, or with a rate outside 1 to 10000, it is nothing.

## The texts

The file texts and the job's fields are read by functions of them alone, so `tests/cpus.cpp` holds them without a real cgroup or job:
`cgroup_v2_cpus(text)` reads `cpu.max` as a quota and a period in microseconds, "max 100000" as no limit (nothing), "600000 100000" as 6 CPUs and "150000 100000" as 2, rounded up; `cgroup_v1_cpus(quota, period)` reads the v1 pair the same way, a quota of -1 as no limit.
A text that is not exactly those fields in decimal, or a zero quota or period, reads as nothing, so a quota or affinity that cannot be read changes nothing.
The mounts are read from mountinfo lines of any number of optional fields, their root and mount point with the kernel's octal escapes (`\040` for a space) decoded.
`automatic_threads(hardware, affinity, quota)` combines the three counts, a zero taken as not read, and the test holds the minimum, the fallback and the bounds there.
