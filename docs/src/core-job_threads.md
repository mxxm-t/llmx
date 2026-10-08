# `src/core/job_threads.hpp` - threads that run a job over its indices

`core::JobThreads` is a fixed set of threads that sleep on a condition variable until a caller hands them a job, and use no CPU between jobs.
`run(n, job)` runs `job(i)` for every index below `n` on its threads and the calling thread, each index once, taken under one mutex as a thread comes free, and returns once every call has returned, rethrowing the first exception a call threw only then, so a job's captures outlive every call; a single index runs on the calling thread without waking a thread.
The calling thread takes the first index and keeps taking indices while any is left, so a job on fewer free cores than threads runs toward the calling thread alone and never waits for a thread that has not woken.
It decides nothing: what an index names, and what a call may touch, are the caller's.

Two callers: the server's scheduler draws a retiring pass's rows on it (`docs/SERVER.md`, Sampling), and a Vulkan tensor group makes each member's queue call of a sum on it (`docs/src/backends-vulkan.md`).
The `job-threads` CTest runs it with no thread and with one, three and four.
