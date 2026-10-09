# Disk tier

A third tier for the server's saved histories, below the device tier and the host tier ([SPECULATIVE](SPECULATIVE.md), section 2, Host tier; [SERVER](SERVER.md)).
What host memory can no longer hold goes to a local disk instead of being dropped, so a conversation that comes back after the host tier has filled reads its history from disk in about half a second rather than recomputing it for tens of seconds.
This page is the plan the tier was built from, kept as agreed; where building changed the design it says so in place (Demotion; Keeping entries across a restart; Entries written as what changed, which replaced the whole copy as the unit on disk), and `docs/STATUS.md` records each step and the figures it was measured with.

## Why

On a long multi-user chat workload on a 27B model a conversation's copy holds hundreds of MiB and a message boundary 150 MiB, so the default host tier, half of what the host has free, holds a few conversations and a few boundaries each ([STATUS](STATUS.md), the edited-turn gap after message boundaries, has the workload and its figures).
Whatever falls out of host memory is recomputed, and the edited-turn gap is mostly boundaries the host tier had no room for.
A disk read of an 800 MiB copy takes about 0.5 s on the MI50 machine (Hardware, below), and the disks hold hundreds of times what host memory does.

## What the host tier does now

Read from `src/server/scheduler.hpp`, `src/model/history.hpp` and `src/server/policy.hpp` at main `6b5c051e`:
- **Entries.** A `HostDonor` is a device donor's history copied to host memory as the devices evict it (`Scheduler::write_back`): its whole KV blocks up to its checkpoint, the checkpoint's state on a model that keeps one, and on a model with an embedded drafter the drafter's carried row with the slot, in a `HostHistory` (per device, runs of 64 MiB host-visible slabs, with tickets for the copies in flight), beside its tokens and row classes.
  A `Boundary` is the state alone at a re-prefill job's checkpoint, or where a request's last user message starts (`Scheduler::keep_boundary`, `Model::save_host` without blocks), forked later with the blocks of a later history of the same conversation (`Model::fork` with a state); with a disk tier the host tier thins no conversation's boundaries, and its room sends the older ones to disk.
- **Copies.** `Model::save_host` and `Model::restore_host` enqueue the copies on each device's stream and return; nothing waits for them, and `release_host` waits for their tickets before the slabs are reused.
- **Room.** `host_cap_` (`--host-cache-bytes`, by default `--max-seqs` histories at the most one request may hold, within half of the free memory after loading, `server::default_host_cache`) bounds the slabs; each slab allocation must leave the host the reserve the fit keeps (`detail::host_room`, `CpuBackend::host_reserve`, a twentieth of what is free).
- **Ranking**, in `write_back` and `keep_boundary`, the host tier's one owner: superseded copies go first and are never copied (a re-prefill job's donor supersedes its conversation's earlier copies, `Scheduler::supersede`); then boundaries, those of the conversation longest unheard first; then copies, where a donor whose conversation did not come back takes only free room and the room of copies that did not come back either, until the tier has refused as many in a row as it holds (`host_refused_`); a copy renewed by a donor of the same history takes that donor's standing.
- **Matching.** `enter` compares the device donors (`best_donor`), the host donors (`best_host`, promoted first by `promote`) and the boundaries (`best_bound`) by tokens and row classes and forks the longest whole-block share; a promoted host entry stays in host memory, renewed, so evicting it again copies nothing.

## What goes to disk

The same histories, as what changed (Entries written as what changed, below): a host donor's blocks as segments and each state as a file of its own, each named by digests of its history's tokens and row classes.
- **Bit-exact.** An entry holds the bytes the storages held, as `save_host` copied them, never converted, compressed lossily or requantized; restored, it gives the bytes of a history never evicted, so "same prompt, same result" holds across all three tiers, as it does across two now.
- **Only from the host tier.** An entry reaches disk only by leaving host memory, and comes back only through host memory: disk to host slabs, then the existing promotion to the devices. The devices never read or write a file, and no new device copy path is added.
- **Paused requests** leave donors like finished ones, so they reach disk the same way; a paused request's own donor is matched by its id as now.
- **Not stored:** anything a request is still using, drafts and marks (section 3 of SPECULATIVE), logits, sampler state.

## The entry file

One file per entry, written once and never changed in place.
- **Header**, fixed layout, its own CRC32C:
  - magic and format version (`DiskStore::kVersion`, 2 since a file is a segment or a state);
  - identity (below), whose every field must equal the running server's for the entry to be read;
  - what the file is (`DiskIndex::describe`): a segment's token range or a state's position, the digest of the history below it and at each of its blocks' ends, and whether its conversation came back, so the index is rebuilt from headers alone; the tokens themselves are not written;
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
- **When.** Just ahead of need: whenever the host entries not yet on disk hold more than three quarters of the host tier and no write is in flight, the entry the host tier would drop next goes to disk, one at a time, while it stays a host entry, readable and promotable. Each write is one file, what the entry's history lacks on disk: its first segment not there, then the next, then its state (`DiskTier::write_next`). Once the last is in place the entry is marked on disk, and room the host tier needs later releases it at once (below).
  Demoting only at the moment of need keeps nothing: the entry handed to the writer still holds its slabs, so the room the copy needs is never freed, and the room order below cancels that very write each time the tier is full. Writing ahead costs the writes of entries later promoted or superseded instead of dropped; superseded ones are deleted at once, promoted ones keep their file, and the writes stay bounded by what passes through the host tier's last quarter (Hardware, wear). Approved so, with the cost counted: `/v1/health`'s `disk_bytes_written` is every byte written ahead and `disk_bytes_read` every byte read back for a request, the disk's wear against its use.
- **Which.** The host tier's ranking (Ranking, above) chooses which entry goes next; the disk tier only decides whether the chosen entry is kept:
  - a superseded copy is never written (it was never worth host room either);
  - a boundary is written, and so is a copy whose conversation came back;
  - a copy whose conversation never came back (one-time conversations, Age, below) is written only into free disk room and the room of other such entries, and is the first to go, the host tier's come-back rule carried down;
  - a history whose files are on disk already, one read from disk earlier or one another conversation wrote, is not written again: its last file is renewed, as a host entry promoted to the devices is renewed now.
- **How.** The scheduler hands the entry's slabs to the writer thread, which writes them to a temporary file chunk by chunk and renames it into place; the entry stays in the host tier's index meanwhile.
  The scheduler thread enqueues the write and returns; it never waits on a file.
- **Writes never block admission.** Room the host tier needs now, for a device donor's write-back, a promotion or a read from disk, is taken in this order, and no step waits for a write:
  1. entries that need no write: superseded copies, and entries whose history is whole on disk, which are simply released;
  2. the host tier's ranking among the rest, each chosen entry dropped as without a disk tier, the entry whose write is in flight last among those the step may take (`Scheduler::written_soon`): the writer writes what the ranking would take next, so taking it first cancelled every write once the writer had fallen behind, and nothing reached the disk again;
  3. that entry, dropped when no other is left, cancels its write: the writer stops at its current chunk, removes the temporary file, and the entry is not kept, as an entry dropped without a disk tier is not. Its slabs return as the chunk in flight ends, at most 4 MiB, a few milliseconds at the measured rates; the room is counted free at once, and the copy that needed it may take new slabs past the host tier's cap by the dropped entry's until they return, so nothing waits on the scheduler thread.
  A demotion that is still writing never delays an admission: the worst it costs is an entry not kept, which is what the host tier does without a disk tier.
- **Write failure** (disk full, I/O error, file system read-only): the entry is dropped as today, the tier stops writing (below), and serving goes on.

## Disk eviction and room

- **Size cap.** `--disk-cache-bytes` bounds the bytes of entry files, temporary files included.
- **Free-space floor.** Before every write the writer reads the file system's free space (`statvfs`, `GetDiskFreeSpaceExW`) and writes only if the file system keeps at least the floor after it (`--disk-cache-floor`); with several servers on one disk the floor is the same physical free space for all of them, so together they never fill it.
- **Order** (`DiskIndex::victim`): only a file nothing on disk stands on goes, so a history loses its end before its base. First a history's earlier states, those with a state of it both below and above them, the least recently used first; then the files of conversations that did not come back, then the rest, each by its conversation's last use. A file a read is on, or that the file being written stands on, stays, and a history a later turn supersedes is that turn's base and stays.
  A state whose history's blocks are not all on disk stays while the server runs, a history in memory holding its rows, and is dropped at the next start.
- **Disk full and I/O errors.** `ENOSPC`, `EIO`, `EROFS` or any failed write ends the tier's writing until the server restarts or the free space recovers above the floor plus the cap's tenth on a later check (every minute); reads go on while they succeed. A read error or a checksum failure deletes that entry and the request computes its history as if the entry had never been there. Counted in `/v1/health` (Surface, below).

## Restore: disk to host to devices

- **Index in memory.** Every file's range and digests, 32 bytes a block, stay in memory (`DiskIndex`), so matching reads no file: a request's history is digested block by block and walked down the tree (`DiskIndex::path`), which shares a file exactly where tokens and row classes are the same.
- **Prefetch on submit.** When a request is submitted, not when it is admitted, the scheduler looks for a path on disk sharing more whole blocks than every device and host entry; if there is one, the read of its files is queued at once, so the read overlaps the request's wait in the queue.
- **Reading.** A reader thread reads the path's files in order with direct I/O, each into its blocks' places in host slabs taken as a promotion takes room (by the order under Demotion, never waiting for a write), verifying each chunk; then the entry is a host entry like any other, and the request's admission promotes it to the devices as now.
- **While it loads** the request is not admitted, but it keeps its place: requests behind it that fit may be admitted past it, the one exception to admission by arrival, since admitting it now would recompute what is about to arrive. Once the entry is in host memory it is admitted first.
- **Bounds.** At most two reads in flight and at most `--max-seqs` waiting; a read waits at most as long as recomputing its tokens would take at the measured prompt rate, from passes of 64 prompt rows or more, and without a bound until a pass has measured one; past that, or on any error, the request is admitted without it; a cancelled request's read completes and leaves a host entry.
- **An entry larger than the host tier**, as a long conversation's copy is under the tier a host short of free memory was given at start (a 76k-token conversation of a 27B model is 5.2 GiB), is read through host memory beyond the tier (`Scheduler::start_read`): nothing is dropped for it, the host's free memory with its reserve decides, as for every slab, and where that refuses the server says the entry was not read and why. It counts in no room, nor do its slabs count against the limit the tier's own copies are allocated within (`Scheduler::parked_held`), so a boundary read beside it is not refused; it is released once no request waits that could still promote it, its slabs freed rather than left for the tier's next copy (`Model::trim_host`), and its file stays.
- **Holders.** A copy in host memory holding a boundary's rows becomes the newest as the boundary's read starts, so the room the read takes goes to other entries first.
- **States** are read with the blocks of their path, and alone where a history on the devices or in host memory holds the rows (`Scheduler::rows_held`), which the read then becomes a boundary beside.

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
- histories whole on disk are not written again; superseded copies in memory are never written, and the files of a superseded history stay as the base of the one that superseded it;
- then, newest first within each class (`Scheduler::flush_one`): the histories whose conversations came back, a device donor's before host memory's entries before the boundaries' states, then those of conversations that never came back;
- of a device donor only the blocks not on disk are copied off the devices, a segment at a time into slabs of the tier's own (`Model::save_host_blocks`), and then its state; a host entry's are written from its own slabs;
- within the cap and the floor, and within a bound the server states before its first write, after which the flush stops at its current entry, removes that entry's temporary file and exits: what is not written is not kept, as with a crash.
**The bound.** It is 20 seconds (`kDiskFlush`), or, where that is longer, half as long again as the bytes still to write would take at the store's measured write rate, plus five seconds for the copies off the devices (`Scheduler::flush_disk`); the rate is the start's probe and then each entry written (`DiskStore::write_rate`).
The server prints the bytes and the bound as the flush starts (`server: writing 5194.0 MiB to disk for the next server, within 20 s`) and, as it ends, the entries kept and the bytes it did not write, with the reason where there are any: the bound reached, or writing having stopped.
A container or service manager must give the server at least the bound before it kills it, and the bound is not a flag: `docker stop` waits 10 seconds by default, so a server under keep runs with a stop timeout of 30 seconds or more, and more where the first line names more; a kill before the flush ends behaves as a crash.

**Written ahead while idle.** A stop should find little to write, so under `--disk-cache-keep` a server with no request active, queued or paused for five seconds (`kDiskIdle`) writes what the flush would, one file at a time (`Scheduler::disk_round`, `flush_one`), in the flush's order; a history whole on disk is not written again.
The slabs a device donor's segment is copied into are the tier's own, beyond the host tier's bytes and at most a segment's (`DiskTier::held`), given back as its file lands, so a long conversation is written whatever tier the host was given.
A request that arrives meanwhile waits for the copy off the devices in progress, as it waits for an eviction's, and its room takes the write in flight last (Demotion, above).

**Every turn, what it added.** A turn's idle write is its new blocks and its state (Entries written as what changed, below), so no rule spaces the writes out: the rule that left a conversation alone until it had grown by a third went with the whole copy it was for, and a crash loses what was not yet written, a turn at most.

**A console closed on Windows.** Windows ends a process a few seconds after its console window is closed, whatever its handler does, so a flush that closing the console starts may not finish and keeps only the entries written by then; Ctrl-C and Ctrl-Break give the flush its whole bound, as SIGTERM and SIGINT do elsewhere.

**SIGTERM against a crash.** SIGTERM and SIGINT take the clean exit and the flush. SIGKILL, a crash, an out-of-memory kill or a power loss keep only what is already on disk: every renamed `.kv` file is complete and every `.tmp` file is removed by the next sweep; the kernel releases the lock, so the directory is adoptable with `--disk-cache-keep` and removed without it.

**What a restart adopts.** A server started with `--disk-cache-keep` adopts, from unlocked directories under its root, the entries whose identity matches:
- it reads every entry's header, after a clean exit and after a crash alike, which holds the file's kind, whether its conversation came back, its range and its digests, and takes its last use from its file's modification time, which every write and renewal sets to the entry's last use; no separate index file is written, the headers being one;
- entries older than the age limit (Age, below) are deleted, and so are the files no whole path from an empty history reaches: a segment whose base is gone, a state whose blocks are not all there;
- the rest are taken within its own cap, the files over it going by the room's order (Disk eviction and room, above);
- the index is rebuilt in memory from what it took, so the first request of a returning conversation matches it as it would have before the restart, and the server's line gives the entries it took and, where it dropped any, how many by each rule: a description that did not read, files no whole path reaches, the cap (`DiskTier::adopt`).
Two servers starting together under one root each adopt only directories whose lock they get; a directory one adopts is moved into its own and is gone for the other.

## Age

Many conversations are used once, and nothing should be kept forever.
- **The age limit**, `--disk-cache-max-age`, by default 24 hours: an entry whose last use, its writing or its latest renewal or promotion, is older than that is deleted, by the periodic sweep every ten minutes and at adoption, in both by what the files hold and never by a file's own age: a file goes only when nothing beyond it was used within the limit, so a conversation used since keeps the older files it stands on, across a restart too, where the store adopts every file and the index then applies the limit (`DiskTier::adopt`). A day covers a conversation picked up again after a meeting, in the afternoon or the next morning, which is when a disk read saves the most recompute; past it, a conversation is more likely abandoned than resumed, and its file holds a conversation's caches, which the privacy default would not keep. The value takes a unit: `90m`, `24h`, `7d`; `0` turns the limit off.
- **One-time conversations** are the copies whose conversation never came back, the host tier's `back` flag, carried in the header and the index. They are written to disk, because a user who comes back hours later to a single long first message, a 20k-token document say, would otherwise wait about 90 s of recompute against a few seconds of reading; but only into free disk room and the room of other one-time entries, and they are the first to go, the host tier's come-back rule carried down, so they never push out a conversation that came back. A one-time entry that is read back becomes a conversation that came back.
- **Wear.** A one-time entry costs one write, once, and only when the host tier loses it; the ones a busy server never needs are overwritten by room long before the age limit, and the cap bounds what they can take. A write budget stays a later option, as planned under Hardware.
- **Across restarts.** The age runs on the wall clock from an entry's last use and is kept in the index and the files' modification times, so a restart neither resets nor pauses it: an entry written an hour before a clean exit and adopted a day later is deleted at adoption; the flush records last uses as they were and does not renew what it writes.

## Privacy

- Directories 0700 and files 0600 on POSIX; on Windows the server sets no permissions and checks no ACL, so its directory and files inherit the ACL of `--disk-cache-dir`, and that directory is where access is decided (the default is under the user's profile).
- File names carry no tokens or text; the files hold no token ids since version 2, only digests of them, and caches from which a conversation could be reconstructed, so they are as private as the conversations.
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

Until the digest and the store's probe finish, the server serves without the disk tier, writing and reading nothing, and `/v1/health`'s `reuse.disk.now.ready` stays false.
**`/v1/health`**, under `reuse.disk`: `now.entries` (files: `now.segments` and `now.states` together), `now.bytes`, `now.limit_bytes`, `now.in_flight`, `now.ready`, `now.writing` (false once the tier has stopped writing), and `since_start.hits`, `.bytes_written`, `.bytes_read`, `.waits` (requests that waited for a read) with `.wait_ms`, `.errors`, `.lost_before_written` (copies room took from host memory before any file held them, each a conversation the tiers lost) and `.dropped_for_cap` (entries deleted while running to stay within the cap; those a start drops for it are in its adoption line).

**Tests:**
- the store alone (CTest): an entry written and read back bit for bit; each identity field changed refuses it; a flipped payload byte fails its chunk and deletes the entry; a truncated file and a `.tmp` file are never read; the cap and the floor stop a write; an injected `ENOSPC` and `EIO` stop writing and keep reads;
- the sweep: a directory whose lock is free is removed, one held by a child process that keeps its lock is untouched, adoption with `--disk-cache-keep` keeps matching entries and removes mismatched ones;
- the scheduler (`server-resume`): on the dense model and the hybrid one with boundaries, a conversation demoted through host memory to disk and asked for again gives the bytes of a fresh model, on one CPU and over a split; a request waiting for its read lets a later request pass and is admitted first once the read is done; a read that fails or exceeds its bound computes the history; a superseded history's files stay as its successor's base; a conversation of four turns writes each block once, held to the byte, and a restart forks it whole and an edit at its third message; with every host slab held by entries being written through a writer slowed to a chunk a second, a request needing host room is admitted without waiting for any write, the room coming first from entries already on disk and then from the newest write, cancelled, whose temporary file is gone and whose entry is not kept, every reply its reply alone;
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

Built after these, 2026-10-08: entries written as what changed (below).

## Entries written as what changed

The user, 2026-10-07: disk writes should write what has changed. Built so in steps (`docs/STATUS.md` has each).
Nothing in it is specific to a backend: the copies run through `Model::save_host`, `Model::save_host_blocks` and `Model::restore_host`, which reach a device only through `Backend::alloc`, `Backend::copy`, `Backend::submit`, `Backend::wait`, `Backend::sync` and `Buffer::host_ptr`, and what makes two servers' bytes interchangeable is `Backend::identity` and `Backend::kv_layout`.

### What changes in a turn, measured

Qwen3.8-27B Q8_0 over two MI50s, a conversation of 76k tokens, turns of about 60 tokens (the ten-turn run of the idle rewrite's record in STATUS):

| | bytes | written and flushed on the test machine's pool |
|---|---|---|
| one KV block, 64 tokens, every layer, K and V, both cards | 4.25 MiB (68.0 KiB a token) | 0.01 s |
| the recurrent state at one position | 149.6 MiB, whatever the length | |
| what a turn of up to 64 new tokens changes: one block and the state | 154 MiB | 0.13 s |
| the whole copy at 76160 tokens | 5207 MiB | 4.8 to 5.2 s |
| 200 files of 6 MiB, written and flushed one by one | 1.2 GB | 2.1 s, 10 ms a file |

Two facts decide the shape. The KV rows of a history are append-only: a later turn's history is the earlier one's whole blocks and more, block for block the same bytes, since a turn forks the turn before. The state is not: it is dense, the same size at every length, and every byte of it differs a turn later. So what changed in a turn is its new blocks, a few MiB, and one state, 150 MiB on this model; no layout makes the state smaller. The gain is the 5.2 GB: never written twice, neither every third of growth nor at the stop.

### The shape on disk

Two kinds of file, both immutable, both written as the store writes any entry (a temporary name, flushed, renamed, the directory flushed):

- **A segment**: the blocks of a token range `[a, b)` of one history, every device, layer by layer, K then V. A segment never crosses a multiple of 1024 tokens (`DiskIndex::kSegmentTokens`): it bounds what a fork copies at 68 MiB on this model and leaves 75 files for 76k tokens.
- **A state**: the recurrent state at one position. A model that keeps no state writes none.

A history of `n` tokens is on disk when segments cover `[0, n)` and, on a model that keeps a state, a state at `n` is there. A message boundary is an earlier state of the same history.

Not chosen: fixed files of one block or one group each, with the last partial one rewritten as it fills. It needs no chain, but rewrites each block up to sixteen times within its group at short turns, where a segment a turn writes each block once. Also not chosen: one file a conversation, appended in place, since an append that tears leaves a file whose end is not known to be whole, and every rule here rests on files that are whole or absent. No index file is kept: the headers are the index.

### Naming a segment: the prefix digest

A file's header carries its range, the digest of everything below it and the digest at each of its blocks' ends (`DiskIndex::describe`); a digest is a SHA-256 chained block by block over the block's tokens and the class each of its rows was computed in (`DiskIndex::digests`). As built the header holds no tokens: a request's own tokens give its digests, and equal digests are equal tokens and classes. Two histories that share a prefix share its digests, so a segment belongs to every history whose tokens and classes begin with its prefix, with no reference counts: the segments on disk form a tree, and a history is a path from the root. A state names the digest at its position.

- **A conversation that goes on** adds segments at the end of its path and one state.
- **An edit or a regenerate** forks at a whole block `p`. Segments wholly below `p` are shared as they are. The one segment that crosses `p` is not cut: the new history writes its own from that segment's start (`DiskIndex::missing`), so it copies at most the blocks of one segment below `p`, under 68 MiB, which is why segments are capped. The other branch stays a path of its own until room or age takes it.
- **Superseding goes** for blocks, a longer history being its shorter self and more.

### Reading a history back

A request that wants `[0, n)` reads the segments on its path in order into the host slabs, each run landing at its blocks' offset in the layer's region, then the state (`DiskTier::read`). A segment the request parts from inside is read whole, its further blocks into slabs of the read's own that are given back as it ends. A request that forks an earlier position reads only the segments below it.
A path reads as fast as the one file a conversation that was: 76 files of a 76k-token conversation cost 0.16 s of a 6.8 s read, their headers, and the read's time is the slabs taken (2.0 s for 5.2 GiB), the disk (3.1 s), and the checksums and the copy into the slabs (1.5 s) (`docs/STATUS.md` has the profile).

The store keeps moving bytes only: it is given, for each file, the spans of host memory its runs go to or come from (`Model::host_ranges`, `Model::host_state_ranges`). Which files make a history, and in which order, is the index's.

Which path a request takes is a walk down the tree (`DiskIndex::path`): from the root, the segment that continues the request's digests, as far as they do, and then the highest state at or below that point on a model that keeps one. Two conversations that begin with one system prompt share its segments. Classes decide that sharing as much as tokens: the same tokens computed as rows of another class are another path, since a fork gives the same bits only over rows of one class.

### What "on disk" means to the host tier

The host tier asks whether a history is on disk, and means the whole of it: its path covers its length and, on a model that keeps a state, its state is there (`DiskTier::on_disk`). `release_written` drops a host copy as one that loses nothing only then, an idle write skips a history only then, and the write-ahead above three quarters counts a host copy as unwritten until then. A history whose last segment or whose state is still to be written is not on disk for any of them. A boundary, which holds a state alone, is on disk once its state is (`DiskTier::state_on_disk`).

### Identity, checksums, version

Every file carries the identity, its own header CRC and a CRC32C for each 4 MiB of payload, and `DiskStore::kVersion` 2. The identity text names the version too (its `entries` line), so a directory a server of version 1 left is not adopted: the first server of this layout starts with an empty cache, once, and names the component that differs.

### Crash safety, case by case

Every file is whole or absent, so the cases are about which files exist together:

- **A crash during a write**: a temporary file, removed by the next sweep.
- **Blocks without their state** (the segments of a turn landed, the state did not): the path is usable up to its newest state at or below the blocks, and the extra blocks are kept, the conversation's next state making them useful.
- **A state without its blocks**: a boundary's state is written whatever of its history's blocks is on disk, since a history in memory holds the rows and whoever holds the blocks writes them. A running server forks such a state with the rows in memory; a start drops it.
- **A gap in a path** (a segment deleted, unreadable or failing its checksum): the segments above the gap are unreachable and are deleted as the gap is found; the path below stays.
- **Adoption** reads every header, builds the tree, and drops, counted by rule in its line: files whose description does not read, files no whole path reaches, then what is over the cap.
- **A path whose newest state is missing** serves from the state below it: the restart forks that one and reads the turns above it again.

### Room, the cap and the age limit

The unit that is deleted is a file, but only ever a leaf: a segment that no segment and no state on disk stands on, or a state. So a path loses its end first and never its base, and a shared prefix goes only after every branch on it.

- **Order** (`DiskIndex::victim`): a history's earlier states first, the least recently used first, never its first nor its newest; then whole conversations, those that did not come back before those that did, the least recently used first, each from its leaf down to where another path joins. A file of a conversation that did not come back takes no room of one that did.
- **Age**: a use renews the path's leaf; a node is as old as its newest descendant, so a base outlives its branches' uses without being touched. An expired conversation goes leaf first.
- **Every state is kept** (the user, 2026-10-07: fast edits remain): each message boundary's state stays on disk, bounded only by the cap and the age limit, so an edit or a regenerate of any earlier message forks the state where that message starts and reads only the message again. The cost is the state a turn, about 150 MiB on Qwen3.8-27B, 7.5 GB for 50 turns and 30 GB for 200 beside 5 GB of blocks at 76k tokens; thinning them to the newest and four was considered and not built.
- **When the cap bites**, within one conversation, the states go first but for two that go last: the newest, which the next turn forks, and the first, where an edit of the opening message forks. Blocks go only as leaves, after the states that stand on them, so each path's base goes last of all.

### The idle write and the stop

Five idle seconds after a turn the server writes, for each history not wholly on disk, the segments past what its path covers and then its state, one file at a time. The copy off the devices is of those blocks alone (`Model::save_host_blocks`), so an idle write of a turn costs its bytes. The stop's flush is the same walk, and what it finds after an idle moment is nothing, and without one the last turn. Its bound counts what it will write, the segments not yet on disk and one state a history, so its first line stays true and its last still says what was not written.

The host tier keeps whole copies in memory as before; when one is written, only the ranges not on disk are.

### Faults, counters and lines

Held by tests since step 4 (`server-resume`'s fault and cap cases, `disk-index`'s age case):

- **A file lost or damaged between two servers** leaves the usable prefix and nothing else, and the restart serves: a segment gone from the middle of a path or its first segment, a temporary file a crash left, a byte flipped in a segment's payload, and, on a model that keeps a state, its newest state or every state gone. A start keeps exactly the files a whole path from an empty history reaches; the next turn forks what they give, and gives the reply of a fresh model.
- **A flipped byte** is not seen by a start, which reads headers; the read finds it, deletes the file and the segments above it, and the request computes its history. The states above stay until the next start, since the blocks computed again are written again under them. A read counts one error however many of its files fail after the first.
- **The cap** never takes a file another stands on: a turn whose segment finds only its own base to delete is not written, and nothing is deleted for it. The stop then says what it did not write and that room was refused.
- **The age limit** takes an expired conversation's own files and leaves what a younger branch stands on.
- **Counters**: `/v1/health` gives `reuse.disk.now.segments` and `.states` beside `.entries`, their sum. When an idle server's writes end it prints what went to disk since they last ended and the files there (`server: 157.2 MiB written to disk since the idle writes last ended; 78 files on disk, 2 of them states`).

Not built, and why: an order that takes a regenerated reply's old tail before other conversations' files. The index cannot tell such a tail from a conversation that shares a prefix with another, a system prompt say, and taking the older of two such conversations first would be wrong; the tail goes by its own last use, as every leaf does.

### Planned: a start with thousands of files

If a start with thousands of files is slow, the header's first page can carry what the tree needs, so a start reads a page a file. Not built: 90 files were in the index 0.14 s after the store was made.

### Decided in review (2026-10-07)

1. The cap of 1024 tokens a segment stays, a constant with its two numbers beside it; the ten-turn run reports the file count of a conversation of many short turns before anyone tunes it.
2. Segments are not folded: it would write each block a second time to save a header and 2 MiB a segment, and read time does not need it.
3. Every state stays on disk within the cap and the age limit (the user, 2026-10-07), at about 150 MiB a turn on Qwen3.8-27B.

## Decisions (agreed with the user 2026-10-04)

Each with the answer the user agreed to:
1. Whether entries should outlive a restart by default once identity is checked, which suits a production restart of the same build, against the privacy default of keeping nothing.
   **Agreed:** keep nothing by default; a production restart passes `--disk-cache-keep`, which flushes memory at a clean exit (Keeping entries across a restart), and the age limit holds across restarts (Age).
2. Whether the model file's full SHA-256, about twenty seconds for a 27 GB file on the MI50 machine with the SHA extensions (1.4 GB/s on its EPYC 7262, against 236 MB/s without them, measured in step 1), should be replaced by a hash of its header and tensor table with its size and modification time, which a file changed in place with the same metadata would defeat.
   **Agreed:** keep the full SHA-256, computed once on a background thread and cached in the sidecar by the file's path, size, modification time and inode, the tier idle until it is known; twenty seconds once a file is acceptable.
3. The default floor and the directory default on a machine whose home directory sits on a small system disk.
   **Agreed:** the tier is off by default (`--disk-cache-bytes 0`), so the operator chooses the directory; at startup the server prints the directory, the file system's free space and the floor, and refuses a start where the cap plus the floor exceed the free space, naming the numbers.
