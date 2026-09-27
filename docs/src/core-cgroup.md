# `src/core/cgroup.hpp` - the cgroups over a process and where their files are read

On Linux a process's limits, a container's among them, are set on the cgroups it lies in, and a limit on any cgroup above the process's holds it too.
`core/cgroup.hpp` finds those cgroups' directories, the one owner of that reading; what is read in them is its readers': the CPU quota's ([cpus](core-cpus.md)) and the memory limits' ([host memory](core-host_memory.md)).
Everything here is a function of the text of `/proc/self/cgroup` and of `/proc/self/mountinfo`, so the tests hold it without a real cgroup; `detail::read_text` gives a file's text on Linux, and a reader passes those two files' texts and reads the files it names through it.

## The directories

`cgroup_v2_directories(proc_self_cgroup, mountinfo)` gives the directory of the process's cgroup v2 and of each cgroup above it, up to its mount's point, its own first.
`cgroup_v1_directories(proc_self_cgroup, mountinfo, controller)` gives the same for its v1 cgroup of `controller`, `cpu` or `memory`, in the hierarchy mounted with that controller.
Either is empty when the process has no such cgroup, the hierarchy is not mounted or no mount holds the cgroup, so a reader over them reads nothing and a limit it cannot find changes nothing.
A reader takes the smallest limit over all of them; a hybrid machine, v1 controllers beside a v2 hierarchy, is read both ways, and the files that exist give the limit.

## Paths and mounts

For cgroup v2 the process's path is its `0::/path` line in `/proc/self/cgroup` (`cgroup_v2_path`) and the mounts are the `cgroup2` ones in `/proc/self/mountinfo` (`cgroup_v2_mounts`); for v1 the path is the first line whose controllers list `controller` (`cgroup_v1_path`) and the mounts are the `cgroup` ones whose super options list it (`cgroup_v1_mounts`).
A path, or a mount's root, that is not absolute or climbs with `..` is not read, and trailing slashes are taken off.
The mounts are read from mountinfo lines, their root and mount point with the kernel's octal escapes (`\040` for a space) decoded; a line is read up to 32 fields, room for 22 optional fields, and a longer line is passed over.

`/proc/self/cgroup` names the path in the cgroup hierarchy as the process's cgroup namespace sees it, which in a container without its own namespace is the path on the host, while a mount can hold only a part of the hierarchy: mountinfo's fourth field is the hierarchy's directory mounted there.
`cgroup_directory` takes the first mount whose root holds the process's path, and the path below that root is its directory under the mount point; a path no mount's root holds is not read.
So a v1 container's cgroup `/c` mounted at `/sys/fs/cgroup/cpu,cpuacct` is read at the mount point, a child cgroup `/c/inner` under it at `/sys/fs/cgroup/cpu,cpuacct/inner`, and a cgroup inside the container that happens to share a name with a directory on the host's path is never taken for the process's own.
From that directory each cgroup is read up to the mount point; cgroups above the mount's root are not visible there and are not read.

`tests/cpus.cpp` (CTest `cpus`) holds the paths, the mounts and, through the CPU quota, the directories on file texts; `tests/host_memory.cpp` (CTest `host-memory`) holds the directories again through the memory limits, v1's in the `memory` controller's hierarchy beside the `cpu` controller's.
