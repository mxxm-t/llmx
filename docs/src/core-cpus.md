# `src/core/cpus.hpp` - the CPUs a process may use and the automatic worker count

`core::automatic_threads()` is the worker count a pool takes when it is given none, the one owner of that count: the CPU backend's constructor takes it ([cpu](backends-cpu.md)), so `--threads` omitted or 0 means it in every command, `serve` and `bench` included (`docs/USAGE.md`, Threads), and the server's scheduler sizes its sampling threads by it: four, or one fewer than it, whichever is fewer ([server](server.md)).
It is the fewest of three counts, each taken only when it could be read, 4 when none could, and never below 1 or above 64:

- the hardware threads, `std::thread::hardware_concurrency()`;
- the CPUs the process's affinity allows (`affinity_cpus`), below;
- the CPUs the process's CPU quota allows, rounded up (`quota_cpus`): its cgroup CPU quotas on Linux (`cgroup_cpus`), its job object's CPU rate hard cap on Windows (`job_cpus`), below.

A container started with `--cpus 6` on a 16-thread host is allowed 6 CPUs of time by its quota while its affinity still lists all 16, so the hardware threads and the affinity alone start 16 workers that the quota throttles together; the quota makes it 6.
The sources are read each time it is called, which is once per CPU backend and once per server scheduler, so a count changed on a running container reaches the next backend made.

## The affinity

On Linux `affinity_cpus` counts the mask `sched_getaffinity` gives.
The mask is allocated with `CPU_ALLOC` and doubled while the kernel refuses it as too small (`EINVAL`), which it does when it has more possible CPUs than the mask holds, up to 2^20 CPUs.

On Windows it goes by the processor groups the process's threads lie in, not the machine's.
`group_affinity()` gives the one group they all lie in, the process affinity mask in it and the group's active processors, from `GetProcessGroupAffinity` and `GetProcessAffinityMask`, and nothing when they span several groups or a mask cannot be read.
With one group the count is that mask's bits, so a restriction such as `start /affinity` or a job object's affinity holds, and a process on a machine of several groups, which Windows 10 keeps in its primary group, counts that group's processors rather than the machine's.
With several groups the process affinity mask is not given, so the count is every active processor of those groups.
`group_affinity()` is also where the CPU backend's prefill placement takes the process mask from ([placement](backends-cpu-placement.md)).

## The cgroup CPU quota (Linux)

`cgroup_cpus(proc_self_cgroup, mountinfo, read)` reads the quota in the process's cgroups, which [cgroup](core-cgroup.md) finds from the text of `/proc/self/cgroup` and `/proc/self/mountinfo`, through `read`, which gives a file's text or nothing; `cgroup_cpus()` passes the real files and reader, and returns nothing off Linux.
For cgroup v2 it reads `cpu.max` in each of `cgroup_v2_directories`; for v1 it reads `cpu.cfs_quota_us` and `cpu.cfs_period_us` in each of `cgroup_v1_directories` for the `cpu` controller.
Those are the process's own cgroup and each above it up to its mount's point, and the smallest limit found wins, since a quota on a parent holds every cgroup under it; a hybrid machine, v1 controllers beside a v2 hierarchy, is read both ways, and the files that exist give the limit.

## The job object's CPU rate (Windows)

`job_cpus()` asks `QueryInformationJobObject` for the CPU rate control of the process's own job, the one a process-isolated Windows container's `--cpus` sets; a job it is nested in is not read.
`job_cpus(control_flags, rate, processors)` reads it: with the enable and hard cap flags set (`job_rate_enable`, `job_rate_hard_cap`, the values Windows gives them) a rate of `rate` cycles in each 10000 over `processors` active processors, those of every group, is that many CPUs rounded up, so 1250 over 16 processors is 2.
Without both flags, as under a weighted rate or a minimum and maximum rate, or with a rate outside 1 to 10000, it is nothing.

## The texts

The file texts and the job's fields are read by functions of them alone, so `tests/cpus.cpp` holds them without a real cgroup or job:
`cgroup_v2_cpus(text)` reads `cpu.max` as a quota and a period in microseconds, "max 100000" as no limit (nothing), "600000 100000" as 6 CPUs and "150000 100000" as 2, rounded up; `cgroup_v1_cpus(quota, period)` reads the v1 pair the same way, a quota of -1 as no limit.
A text that is not exactly those fields in decimal, or a zero quota or period, reads as nothing, so a quota or affinity that cannot be read changes nothing.
The test also holds [cgroup](core-cgroup.md)'s paths and mounts, and reads the quota over a file system held in a map.
`automatic_threads(hardware, affinity, quota)` combines the three counts, a zero taken as not read, and the test holds the minimum, the fallback and the bounds there.
