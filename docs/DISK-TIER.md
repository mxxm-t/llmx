# Disk tier

A third tier for the server's saved histories, below the device tier and the host tier ([SPECULATIVE](SPECULATIVE.md), section 2, Host tier; [SERVER](SERVER.md)).
What host memory can no longer hold goes to a local disk instead of being dropped, so a conversation that comes back after the host tier has filled reads its history from disk in about half a second rather than recomputing it for tens of seconds.
This page is the plan the tier was built from, kept as agreed; where building changed the design it says so in place (Demotion; Keeping entries across a restart), and `docs/STATUS.md` records each step and the figures it was measured with.

## Why

On the six-user, twenty-turn workload of Qwen3.8-27B Q8_0 on one MI50 ([STATUS](STATUS.md), the edited-turn gap after message boundaries), a conversation's copy holds about 800 MiB at turn 19 and a message boundary 150 MiB, and the default host tier, half of what the host has free, holds about six conversations and four boundaries each.
Whatever falls out of host memory is recomputed: about 4.4 ms a token, so 43 s for a 9.7k-token conversation, and the edited-turn gap is mostly boundaries the host tier had no room for (2.54 s at a 16 GiB host tier against 3.31 s at 10 GiB).
A disk read of the same 800 MiB takes about 0.5 s on the MI50 machine (Hardware, below), and the disks hold hundreds of times what host memory does.

## What the host tier does now

Read from `src/server/scheduler.hpp`, `src/model/history.hpp` and `src/server/policy.hpp` at main `6b5c051e`:
- **Entries.** A `HostDonor` is a device donor's history copied to host memory as the devices evict it (`Scheduler::write_back`): its whole KV blocks up to its checkpoint, the checkpoint's state on a model that keeps one, and on a model with an embedded drafter the drafter's carried row with the slot, in a `HostHistory` (per device, runs of 64 MiB host-visible slabs, with tickets for the copies in flight), beside its tokens and row classes.
  A `Boundary` is the state alone at a re-prefill job's checkpoint, or where a request's last user message starts (`Scheduler::keep_boundary`, `Model::save_host` without blocks), forked later with the blocks of a later history of the same conversation (`Model::fork` with a state); with a disk tier the host tier thins no conversation's boundaries, and its room sends the older ones to disk.
- **Copies.** `Model::save_host` and `Model::restore_host` enqueue the copies on each device's stream and return; nothing waits for them, and `release_host` waits for their tickets before the slabs are reused.
- **Room.** `host_cap_` (`--host-cache-bytes`, by default `--max-seqs` histories at the most one request may hold, within half of the free memory after loading, `server::default_host_cache`) bounds the slabs; each slab allocation must leave the host the reserve the fit keeps (`detail::host_room`, `CpuBackend::host_reserve`, a twentieth of what is free).
- **Ranking**, in `write_back` and `keep_boundary`, the host tier's one owner: superseded copies go first and are never copied (a re-prefill job's donor supersedes its conversation's earlier copies, `Scheduler::supersede`); then boundaries, those of the conversation longest unheard first; then copies, where a donor whose conversation did not come back takes only free room and the room of copies that did not come back either, until the tier has refused as many in a row as it holds (`host_refused_`); a copy renewed by a donor of the same history takes that donor's standing.
- **Matching.** `enter` compares the device donors (`best_donor`), the host donors (`best_host`, promoted first by `promote`) and the boundaries (`best_bound`) by tokens and row classes and forks the longest whole-block share; a promoted host entry stays in host memory, renewed, so evicting it again copies nothing.

## What goes to disk

The same entries, unchanged: a host donor's `HostHistory` and a boundary's state, each with its tokens and row classes.
- **Bit-exact.** An entry holds the bytes the storages held, as `save_host` copied them, never converted, compressed lossily or requantized; restored, it gives the bytes of a history never evicted, so "same prompt, same result" holds across all three tiers, as it does across two now.
- **Only from the host tier.** An entry reaches disk only by leaving host memory, and comes back only through host memory: disk to host slabs, then the existing promotion to the devices. The devices never read or write a file, and no new device copy path is added.
- **Paused requests** leave donors like finished ones, so they reach disk the same way; a paused request's own donor is matched by its id as now.
- **Not stored:** anything a request is still using, drafts and marks (section 3 of SPECULATIVE), logits, sampler state.

## The entry file

One file per entry, written once and never changed in place.
- **Header**, fixed layout, its own CRC32C:
  - magic, format version, entry kind (copy or boundary);
  - identity (below), whose every field must equal the running server's for the entry to be read;
  - the entry's tokens and row classes (`RowClass` stretches), so the index can be rebuilt from headers alone;
  - the layout: per device, the bytes of each run in the order `HostSpan` writes them (each KV storage's blocks layer by layer, K then V, then the state slot, then the carried row), the block tokens, the layers, the K and V block bytes and the state slot bytes;
  - the payload's length and the CRC32C of each 4 MiB chunk of it.
- **Payload**: the runs, device by device, padded to the file system's direct-I/O granule.
- **Identity** (what makes two servers' bytes interchangeable):
  - the model file's SHA-256: from the hub manifest when the file was pulled, otherwise computed once on a background thread at startup and kept in a sidecar keyed by the file's path, size, modification time and inode, the tier reading and writing nothing until it is known;
  - the placement: each device's kind and index and the layers it holds, the KV pools' block tokens, the state slots' shape;
  - each device's effective activation dtype and the K and V cache types;
  - what in the build can change a history's bits, and not the revision, so an update that changes none of it reads the entries of the server before (`disk_identity` in `src/server/disk_tier.hpp`):
    - the numerics fingerprint: the SHA-256 over every file under `src/`, `cmake/`, `CMakeLists.txt` and `build.bat` that `cmake/numerics-sources.txt` does not name out, each file's bytes with CRLF read as LF. A file is in unless a line there argues it out (the CLI, the hub, the server, the tokenizer, and the files that only parse, sample or print), so the backends and their shaders, the model and its architectures, the quantization decoders, the format readers and the loader are in without being named. Both build routes compute it at every build, CMake in `cmake/build-info.cmake` and `build.bat` through `cmake/numerics-fingerprint.ps1`, with no Git, and `tests/version.py` computes it a third time and holds the binary to it;
    - the compiler's own version text, the compiler, configuration, flags and options the build system used, which no file holds, and the shader compiler's version;
    - the C library's version where it names itself (glibc), since the math functions in the position tables and norms come from it; on Windows and macOS it is unrecorded;
    - each device's line of the layout: a Vulkan device's vendor and device ids, its driver's name, info and version and its pipeline cache id, which changes with every build of the driver, also one that keeps its version strings; the CPU's is the word `cpu`, its model unrecorded, the kernels' AVX2 and FMA baseline being one on every machine they run on and a cache directory not moving between machines;
    A server leaves these components as text in its directory (`identity`), prints the fingerprint's first 16 digits in its start line, and where it adopts nothing from a kept directory says which components differ (`server: a kept directory was not adopted, its numerics differing from this server's`), or that the directory was written by a build that kept no identity beside its entries. `llmx --version` prints the fingerprint too.
  - the row-class signature: a hash of `Model::row_class` over every extent from 1 to the token limit, computed once at startup, since the classes are what make a fork exact ([SPECULATIVE](SPECULATIVE.md), section 1, row classes).
- **Checksum.** CRC32C, with the SSE4.2 instruction the AVX2 baseline includes and a table fallback, at many GB/s, so verifying an entry costs far less than reading it; SHA-256 at about 1 GB/s a core would be slower than the disk. The header's CRC is checked by the startup sweep and the index rebuild, the payload's chunk by chunk as an entry is read, before anything reaches host slabs that a request can see.

## Demotion: host to disk

The host tier stays the window of the newest entries and the disk tier keeps what slides out of it.
- **When.** Just ahead of need: whenever the host entries not yet on disk hold more than three quarters of the host tier and no write is in flight, the entry the host tier would drop next goes to disk, one at a time, while it stays a host entry, readable and promotable. Once its file is in place the entry is marked on disk, and room the host tier needs later releases it at once (below).
  Demoting only at the moment of need, as a first version of this plan had it, keeps nothing: the entry handed to the writer still holds its slabs, so the room the copy needs is never freed, and the room order below cancels that very write each time the tier is full. Writing ahead costs the writes of entries later promoted or superseded instead of dropped; superseded ones are deleted at once, promoted ones keep their file, and the writes stay bounded by what passes through the host tier's last quarter (Hardware, wear). Approved so, with the cost counted: `/v1/health`'s `disk_bytes_written` is every byte written ahead and `disk_bytes_read` every byte read back for a request, the disk's wear against its use.
- **Which.** The host tier's ranking chooses which entry goes next: boundaries, oldest first, then copies whose conversations did not come back, oldest first, then the oldest copies; the disk tier only decides whether the chosen entry is kept:
  - a superseded copy is never written (it was never worth host room either);
  - a boundary is written, and so is a copy whose conversation came back;
  - a copy whose conversation never came back (one-time conversations, Age, below) is written only into free disk room and the room of other such entries, and is the first to go, the host tier's come-back rule carried down;
  - an entry already on disk with the same history, one promoted from disk earlier, is not written again: its file is renewed, as a host entry promoted to the devices is renewed now.
- **How.** The scheduler hands the entry's slabs to the writer thread, which writes them to a temporary file chunk by chunk and renames it into place; the entry stays in the host tier's index meanwhile.
  The scheduler thread enqueues the write and returns; it never waits on a file.
- **Writes never block admission.** Room the host tier needs now, for a device donor's write-back, a promotion or a read from disk, is taken in this order, and no step waits for a write:
  1. entries that need no write: superseded copies, and entries whose file is already on disk (renewed ones), which are simply released;
  2. the host tier's ranking among the rest, each chosen entry dropped as without a disk tier, the entry whose write is in flight last among those the step may take (`Scheduler::written_soon`): the writer writes what the ranking would take next, so taking it first cancelled every write once the writer had fallen behind, and nothing reached the disk again;
  3. that entry, dropped when no other is left, cancels its write: the writer stops at its current chunk, removes the temporary file, and the entry is not kept, as an entry dropped without a disk tier is not. Its slabs return as the chunk in flight ends, at most 4 MiB, a few milliseconds at the measured rates; the room is counted free at once, and the copy that needed it may take new slabs past the host tier's cap by the dropped entry's until they return, so nothing waits on the scheduler thread.
  A demotion that is still writing never delays an admission: the worst it costs is an entry not kept, which is what the host tier does without a disk tier.
- **Write failure** (disk full, I/O error, file system read-only): the entry is dropped as today, the tier stops writing (below), and serving goes on.

## Disk eviction and room

- **Size cap.** `--disk-cache-bytes` bounds the bytes of entry files, temporary files included.
- **Free-space floor.** Before every write the writer reads the file system's free space (`statvfs`, `GetDiskFreeSpaceExW`) and writes only if the file system keeps at least the floor after it (`--disk-cache-floor`); with several servers on one disk the floor is the same physical free space for all of them, so together they never fill it.
- **Order**, the host tier's ranking carried down: superseded entries are deleted at once (a copy superseded while on disk is deleted, not kept for room); then boundaries of the conversation longest unheard; then copies whose conversations did not come back, oldest first; then the oldest.
  A boundary whose conversation has no copy left in any tier can never be forked and goes first of all.
- **Disk full and I/O errors.** `ENOSPC`, `EIO`, `EROFS` or any failed write ends the tier's writing until the server restarts or the free space recovers above the floor plus the cap's tenth on a later check (every minute); reads go on while they succeed. A read error or a checksum failure deletes that entry and the request computes its history as if the entry had never been there. Counted in `/v1/health` (Surface, below).

## Restore: disk to host to devices

- **Index in memory.** Every entry's tokens and row classes stay in the scheduler's memory (40 KB for a 10k-token conversation), so matching reads no file; `enter` compares disk entries beside device donors, host donors and boundaries by the same rule (`shareable`).
- **Prefetch on submit.** When a request is submitted, not when it is admitted, the scheduler looks for a disk entry sharing more whole blocks than every device and host entry; if one does, its read is queued at once, so the read overlaps the request's wait in the queue.
- **Reading.** A reader thread reads the file with direct I/O into host slabs taken as a promotion takes room (by the order under Demotion, never waiting for a write), verifying each chunk; then the entry is a host entry like any other, and the request's admission promotes it to the devices as now.
- **While it loads** the request is not admitted, but it keeps its place: requests behind it that fit may be admitted past it, the one exception to admission by arrival, since admitting it now would recompute what is about to arrive. Once the entry is in host memory it is admitted first.
- **Bounds.** At most two reads in flight and at most `--max-seqs` waiting; a read waits at most as long as recomputing its tokens would take at the measured prompt rate, from passes of 64 prompt rows or more, and without a bound until a pass has measured one; past that, or on any error, the request is admitted without it; a cancelled request's read completes and leaves a host entry.
- **An entry larger than the host tier**, as a long conversation's copy is under the tier a host short of free memory was given at start (a 76k-token conversation of a 27B model is 5.2 GiB), is read through host memory beyond the tier (`Scheduler::start_read`): nothing is dropped for it, the host's free memory with its reserve decides, as for every slab, and where that refuses the server says the entry was not read and why. It counts in no room, nor do its slabs count against the limit the tier's own copies are allocated within (`Scheduler::parked_held`), so a boundary read beside it is not refused; it is released once no request waits that could still promote it, its slabs freed rather than left for the tier's next copy (`Model::trim_host`), and its file stays.
- **Holders.** A copy in host memory holding a boundary's rows becomes the newest as the boundary's read starts, so the room the read takes goes to other entries first.
- **Boundaries** load with the copy that holds their conversation's blocks: a boundary whose copy is on disk loads both, the copy first.

## Crash safety and cleanup

- **Directory.** The tier lives in `--disk-cache-dir` (default `<home>/.cache/llmx/kv`, beside the hub cache), in a directory of its own for each server: `server-<pid>-<random>`, created mode 0700, holding a `lock` file the server keeps locked for its whole life (`flock` with `LOCK_EX`, `LockFileEx` on Windows), released by the kernel however the process ends.
- **Writing.** An entry is written to `entry-<id>.tmp`, flushed (`fdatasync`, `FlushFileBuffers`), then renamed to `entry-<id>.kv` and the directory flushed; a reader, the index or the sweep only ever sees complete files.
- **Startup sweep.** Every server, before it writes, tries the lock of every instance directory under the root: a directory whose lock it gets belonged to a server that is gone, and is removed with everything in it, temporary files included; a directory whose lock is held is another live server's and is never touched. Its own directory starts empty.
  The sweep runs again every ten minutes, so a crashed server's leftovers on a disk shared with servers that keep running do not wait for the next start.
- **Clean exit.** The server removes its own directory before it exits, so nothing it wrote remains.
- **Keeping entries for a restart**, `--disk-cache-keep`: on a clean exit the server first flushes what it holds in memory (Keeping entries across a restart, below), then unlocks and leaves its directory; the next server under the same root whose identity matches adopts it, and with a mismatched identity, a new build or another model, the next sweep removes it. A crashed server's directory is adopted the same way when the flag is given, its temporary files removed and every entry's header checked.
- **Killed mid-write.** A `.tmp` file is never read and is removed by the sweep; a `.kv` file exists only after its rename, so a crash can leave a complete entry or nothing, never a torn one; a torn payload from a failing disk fails its chunk CRC on read.
- **Several servers on one disk.** Each has its own directory and cap; the free-space floor is shared physical space; a sweep removes only unlocked directories; two servers never read each other's live entries.
- **Never filled by leftovers.** Leftovers exist only in unlocked directories, which every start and every periodic sweep removes; every write checks the cap and the floor; and with `--disk-cache-bytes 0`, the default, nothing is written at all.

## Keeping entries across a restart

With `--disk-cache-keep`, a restart of the same model on a build of the same numerics (Identity, above) should find the conversations it was serving, the newest included, which until then live only on the devices and in host memory.

**Clean exit.** Today `serve` has no handler for SIGTERM or SIGINT, so either ends the process at once; the plan adds one (on Windows the console's control events), which makes a clean exit:
1. stop accepting connections and admitting requests;
2. cancel the requests in flight once their pass has retired, as a cancelled client is today: a request keeps what its earlier turns left in the tiers, and its partial turn is not kept;
3. with `--disk-cache-keep`, flush (below), then mark the directory kept and unlock it; without it, remove the directory;
4. exit.

**The flush.** Everything worth keeping that is not yet on disk is written, the most valuable first, the reverse of the order in which room takes entries:
- entries already on disk only have their last use recorded; superseded copies, on the devices, in host memory or on disk, are never written and are deleted;
- then, newest first within each class: copies whose conversations came back, then their boundaries, then copies whose conversations never came back;
- a device donor goes through host memory as an eviction does (`Model::save_host` into slabs that entries already written have given back), then to disk; a host entry goes straight to disk; the writer streams entry after entry so the device copies and the disk writes overlap;
- within the cap and the floor, and within a bound the server states before its first write, after which the flush stops at its current entry, removes that entry's temporary file and exits: what is not written is not kept, as with a crash.
**The bound.** It is 20 seconds (`kDiskFlush`), or, where that is longer, half as long again as the bytes still to write would take at the store's measured write rate, plus five seconds for the copies off the devices (`Scheduler::flush_disk`); the rate is the start's probe and then each entry written (`DiskStore::write_rate`).
The server prints the bytes and the bound as the flush starts (`server: writing 5194.0 MiB to disk for the next server, within 20 s`) and, as it ends, the entries kept and the bytes it did not write, with the reason where there are any: the bound reached, or writing having stopped.
A container or service manager must give the server at least the bound before it kills it, and the bound is not a flag: `docker stop` waits 10 seconds by default, so a server under keep runs with a stop timeout of 30 seconds or more, and more where the first line names more; a kill before the flush ends behaves as a crash.

**Written ahead while idle.** A stop should find little to write, so under `--disk-cache-keep` a server with no request active, queued or paused for five seconds (`kDiskIdle`) writes what the flush would, one entry at a time (`Scheduler::disk_round`, `flush_one`): first what host memory holds without a file, then the device donors, newest first, each copied to host memory as an eviction copies it; a donor whose copy is in host memory or on disk already is not copied again.
A donor's copy larger than the whole host tier is held beyond the tier's bytes until its file is in place and then released (`HostDonor::through`), so a long conversation is written whatever tier the host was given; outside the flush and the idle writes such a copy is not made, as before.
A request that arrives meanwhile waits for the copy off the devices in progress, as it waits for an eviction's, and its room takes the write in flight last (Demotion, above).

**A console closed on Windows.** Windows ends a process a few seconds after its console window is closed, whatever its handler does, so a flush that closing the console starts may not finish and keeps only the entries written by then; Ctrl-C and Ctrl-Break give the flush its whole bound, as SIGTERM and SIGINT do elsewhere.

**SIGTERM against a crash.** SIGTERM and SIGINT take the clean exit and the flush. SIGKILL, a crash, an out-of-memory kill or a power loss keep only what is already on disk: every renamed `.kv` file is complete and every `.tmp` file is removed by the next sweep; the kernel releases the lock, so the directory is adoptable with `--disk-cache-keep` and removed without it.

**What a restart adopts.** A server started with `--disk-cache-keep` adopts, from unlocked directories under its root, the entries whose identity matches:
- it reads every entry's header, after a clean exit and after a crash alike, which holds the entry's kind, whether its conversation came back, its tokens and its row classes, and takes its last use from its file's modification time, which every write and renewal sets to the entry's last use; no separate index file is written, the headers being one;
- entries older than the age limit (Age, below) are deleted, and so is a boundary whose conversation has no copy left;
- the rest are taken, the most valuable first by the flush's order, within its own cap and floor, and the remainder deleted;
- the index is rebuilt in memory from what it took, so the first request of a returning conversation matches it as it would have before the restart, and the server's line gives the entries it took and, where it dropped any, how many by each rule: a description that did not read, a boundary without a copy of its conversation, the cap.
Two servers starting together under one root each adopt only directories whose lock they get; a directory one adopts is moved into its own and is gone for the other.

## Age

Many conversations are used once, and nothing should be kept forever.
- **The age limit**, `--disk-cache-max-age`, by default 24 hours: an entry whose last use, its writing or its latest renewal or promotion, is older than that is deleted, by the periodic sweep every ten minutes and at adoption. A day covers a conversation picked up again after a meeting, in the afternoon or the next morning, which is when a disk read saves the most recompute; past it, a conversation is more likely abandoned than resumed, and its file holds token ids the privacy default would not keep. The value takes a unit: `90m`, `24h`, `7d`; `0` turns the limit off.
- **One-time conversations** are the copies whose conversation never came back, the host tier's `back` flag, carried in the header and the index. They are written to disk, because a user who comes back hours later to a single long first message, a 20k-token document say, would otherwise wait about 90 s of recompute against a few seconds of reading; but only into free disk room and the room of other one-time entries, and they are the first to go, the host tier's come-back rule carried down, so they never push out a conversation that came back. A one-time entry that is read back becomes a conversation that came back.
- **Wear.** A one-time entry costs one write, once, and only when the host tier loses it; the ones a busy server never needs are overwritten by room long before the age limit, and the cap bounds what they can take. A write budget stays a later option, as planned under Hardware.
- **Across restarts.** The age runs on the wall clock from an entry's last use and is kept in the index and the files' modification times, so a restart neither resets nor pauses it: an entry written an hour before a clean exit and adopted a day later is deleted at adoption; the flush records last uses as they were and does not renew what it writes.

## Privacy

- Directories 0700 and files 0600 on POSIX; on Windows the directory under the user's profile inherits its owner-only ACL, and the server refuses a `--disk-cache-dir` whose existing ACL grants others access.
- File names carry no tokens or text; the files hold token ids and caches from which a conversation could be reconstructed, so they are as private as the conversations.
- By default nothing remains after a clean exit, and after a crash only until the next sweep; `--disk-cache-keep` is the one way to keep entries past an exit.
- A stronger mode, planned as an option after measurement: entries as files unlinked as soon as they are open (`O_TMPFILE` on Linux, `FILE_FLAG_DELETE_ON_CLOSE` on Windows), so nothing remains after any exit, crash included, at the cost of an open handle per entry and no adoption.
- No encryption at rest in this plan; a disk that must not hold conversations should not be given to the tier.

## Hardware

**Measured on the MI50 machine**, 2026-10-04, `fio` 3.39 on cores 12-15, 8 GiB of incompressible data, 4 MiB blocks, queue depth 8, files removed after; OpenZFS 2.4.1 with lz4 compression and a 32 GiB ARC:

| file system | write, buffered | write, direct | read, buffered | read, direct |
|---|---:|---:|---:|---:|
| `/` (rpool, one NVMe SSD, recordsize 128K) | 328 MB/s | 343 MB/s | 2745 MB/s | 1606 MB/s |
| `/zpool1` (raidz1 of six SATA SSDs, recordsize 1M) | 2053 MB/s | 555 MB/s | 3536 MB/s | 1778 MB/s |

The buffered reads followed the writes within the ARC's size, so they are cache reads; the direct reads are the disks'.
The buffered write to `/zpool1` includes its final flush and still lands in the ARC first; the NVMe pool's writes are its own sustained rate.
So a 27B conversation of 800 MiB is written in 1.5 to 2.5 s on the writer thread and read in about 0.5 s, and a boundary of 150 MiB in 0.3 to 0.45 s and 0.1 s, against 43 s and about 2 s of recompute.

- **Direct I/O against the page cache.** Reads go around the cache (`FileReader` with `direct`, as the loader's direct mode does): the host tier is the cache, and a page-cache copy would take the host memory the host tier's default already counts on. Writes go around it too where the file system takes direct writes, so a demotion does not grow the page cache or the ARC by the bytes it frees; buffered writes, flushed per entry, are the fallback. On ZFS a direct write was a quarter of a buffered one on `/zpool1` and equal on the NVMe pool, so the write mode is chosen at startup by a short probe in the tier's own directory, and the choice is printed.
- **SSD wear.** Writes happen only when the host tier loses an entry, once per entry, never rewriting a renewed one; `/v1/health` counts the bytes written. On the six-user workload the host tier moved about 145 GB a 17-minute run both ways, the disk tier's writes being the part host memory could not keep; a server writing 100 GB an hour writes 2.4 TB a day, which a 600 TBW consumer drive takes in about eight months, so the default stays off, the docs give the arithmetic, and a write budget per hour is a planned option if measurements on production show the need.
- **Placement.** The root NVMe pool is nearly three quarters full on the MI50 machine and takes about 340 MB/s sustained writes; `/zpool1` writes faster and has more room. Neither is chosen by llmx; the operator gives the directory.

## Interface

One store interface behind the host tier's owner, so the scheduler keeps every decision of what is kept, renewed, demoted, promoted or dropped, and a store only moves bytes:
- `put(key, header, slabs)`: write an entry, asynchronously, calling back with success or the error;
- `get(key, slabs)`: read an entry into host slabs the caller provides, verifying it, asynchronously;
- `evict(key)`: remove an entry;
- `free_bytes()`: what the floor reads; the bytes against the cap are the scheduler's index's.

The index (tokens, row classes, ranking state) stays in the scheduler beside `host_` and `bounds_`; the store holds no policy.
**First implementation:** a disk store in `src/server/`, over a direct-I/O file writer beside `format::FileReader` in `src/format/`, the lock and the sweep with it.
**Second, planned and not built:** a remote store, over HTTP blobs, an S3-compatible bucket or a cache server of our own, so servers on several machines can share histories; its latency, bandwidth, authentication and the privacy of conversations leaving the machine need a plan of their own before any of it is built.

## Surface

**Flags** (`serve`):
- `--disk-cache-bytes N`: the disk tier's size; 0, the default, turns it off.
- `--disk-cache-dir PATH`: where it lives, by default `<home>/.cache/llmx/kv`.
- `--disk-cache-floor N`: the free space the file system keeps after every write, by default the larger of 16 GiB and a twentieth of the file system.
- At startup the server prints the directory, the file system's free space and the floor, and refuses to start when the cap plus the floor exceed the free space, naming the three numbers.
- `--disk-cache-keep`: write what memory holds to disk while idle and, on a clean exit, within the bound the server states (20 seconds or more), and keep the entries for the next server of the same numerics and model, which adopts them; without it nothing remains after a clean exit.
- `--disk-cache-max-age TIME`: delete entries unused for longer than TIME, by default `24h`; `0` keeps them until room takes them.
- The tier needs the host tier: with `--host-cache-bytes 0`, or every cache on the CPU where the host tier's default is 0, a nonzero `--disk-cache-bytes` is refused with the reason.

Until the digest and the store's probe finish, the server serves without the disk tier, writing and reading nothing, and `/v1/health`'s `disk_ready` stays false.
**`/v1/health`:** `disk_entries`, `disk_bytes`, `disk_hits`, `disk_bytes_written`, `disk_bytes_read`, `disk_waits` (requests that waited for a read) and `disk_wait_ms`, `disk_errors`, `host_unwritten` (copies room took from host memory before any file held them, each a conversation the tiers lost), `disk_capped` (entries deleted while running to stay within the cap; those a start drops for it are in its adoption line), and `disk_writing` (false once the tier has stopped writing).

**Tests:**
- the store alone (CTest): an entry written and read back bit for bit; each identity field changed refuses it; a flipped payload byte fails its chunk and deletes the entry; a truncated file and a `.tmp` file are never read; the cap and the floor stop a write; an injected `ENOSPC` and `EIO` stop writing and keep reads;
- the sweep: a directory whose lock is free is removed, one held by a child process that keeps its lock is untouched, adoption with `--disk-cache-keep` keeps matching entries and removes mismatched ones;
- the scheduler (`server-resume`): on the dense model and the hybrid one with boundaries, a conversation demoted through host memory to disk and asked for again gives the bytes of a fresh model, on one CPU and over a split; a request waiting for its read lets a later request pass and is admitted first once the read is done; a read that fails or exceeds its bound computes the history; a superseded entry is never written; with every host slab held by entries being written through a writer slowed to a chunk a second, a request needing host room is admitted without waiting for any write, the room coming first from entries already on disk and then from the newest write, cancelled, whose temporary file is gone and whose entry is not kept, every reply its reply alone;
- the age and one-time entries (`server-resume`, with a clock the test drives): an entry past the age limit is deleted by the sweep and refused at adoption; one-time copies take only free room and room of their own kind, go first, and a one-time entry read back counts as come back;
- the server (`tests/server.py`): `llmx serve` killed with SIGKILL while it writes, then a second server under the same root sweeps the first's directory and serves; a clean exit leaves the root empty; with `--disk-cache-keep`, SIGTERM flushes the device donors and host entries, the next server of the same build adopts them and a returning conversation forks its whole history with the reply it gives on a fresh server; a flush cut short by its limit, and a SIGKILL during the flush, leave only complete entries, which are adopted.

**Gates:**
- server tier as for any server change (CTest, the CPU suite, Qwen3-0.6B byte identity, the hosted run), and the device tier if the model side changes;
- the edited-turn workload against main and the reference server at the default host tier with a disk tier, the edit's and regenerate's p50 and p99, with the disk's bytes written;
- many conversations coming back: more users than the host tier holds (24 users of twenty turns on Qwen3.8-27B Q8_0), the follow-ups' time to first token, the reads they took from disk and the conversations recomputed, against main and the reference server;
- the same on `/zpool1` and on the NVMe pool, so the plan's read and write figures are checked against the server's own.

## Order of work

1. **Identity and layout** (`model/history.hpp`, `model/runtime.hpp`): a host history's layout descriptor and the model identity (row-class signature, placement, dtypes, cache types), and the model file's SHA-256 on a background thread with its sidecar; tests of each field. No file is written.
2. **The store** (`src/server/`, a direct-I/O writer in `src/format/`): the interface, the disk store with its entry file, CRC32C, temporary files and rename, the instance directory and its lock, the sweep and adoption; CTest cases of the store alone. Nothing calls it.
3. **Demotion** (scheduler): host entries the host tier drops go to disk by its ranking; the writer thread; the cap, the floor and stopping on errors; the in-memory index; health counters; `server-resume` cases for writing, ranking and superseded entries.
4. **Restore**: prefetch on submit, the reader thread, requests waiting for their reads, bounds and fallbacks; identity after restore on the dense and hybrid models, one CPU and a split.
5. **Exit, keep and age**: the SIGTERM and SIGINT handler and the clean exit, clean-exit removal, the flush, the index, `--disk-cache-keep` and adoption, `--disk-cache-max-age` in the sweep and at adoption, the periodic sweep, permissions; the kill, flush and restart cases in `tests/server.py`.
6. **Measure and land**: the gates above, the docs (USAGE, SERVER, SPECULATIVE's host tier, the `docs/src` pages), STATUS.

Later, each on its own measurement: the unlinked-file privacy mode, a write budget, and the remote store.

## Decisions (agreed with the user 2026-10-04)

Each with the answer the user agreed to:
1. Whether entries should outlive a restart by default once identity is checked, which suits a production restart of the same build, against the privacy default of keeping nothing.
   **Agreed:** keep nothing by default; a production restart passes `--disk-cache-keep`, which flushes memory at a clean exit (Keeping entries across a restart), and the age limit holds across restarts (Age).
2. Whether the model file's full SHA-256, about twenty seconds for a 27 GB file on the MI50 machine with the SHA extensions (1.4 GB/s on its EPYC 7262, against 236 MB/s without them, measured in step 1), should be replaced by a hash of its header and tensor table with its size and modification time, which a file changed in place with the same metadata would defeat.
   **Agreed:** keep the full SHA-256, computed once on a background thread and cached in the sidecar by the file's path, size, modification time and inode, the tier idle until it is known; twenty seconds once a file is acceptable.
3. The default floor and the directory default on a machine whose home directory sits on a small system disk.
   **Agreed:** the tier is off by default (`--disk-cache-bytes 0`), so the operator chooses the directory; at startup the server prints the directory, the file system's free space and the floor, and refuses a start where the cap plus the floor exceed the free space, naming the numbers.
