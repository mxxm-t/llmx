# `src/format/mapped_file.hpp` - read-only file mapping

`format::MappedFile` maps a whole file read-only and unmaps it when the last
holder lets it go (`GGUFModel` holds it through a `shared_ptr`). Windows
opens the file with `CreateFileW` from its UTF-8 path and maps it with
`CreateFileMappingW` and `MapViewOfFile`; elsewhere it is `mmap` of a
private read-only mapping, the descriptor closed once mapped. An empty file
maps to no pages.

The pages are the operating system's to evict and read back, which is why
the GGUF reader maps each file, including each shard, instead of copying its
payload into the heap: a
model larger than what the host can hold beside it still loads, and the
pages of weights a device has copied are not touched again. Qwen3-30B-A3B
Q8_0 (32.5 GB) with thirty layers' experts on a 32 GB host paged through
every pass when it was one heap allocation.

`drop(p, bytes)` tells the OS that the whole pages inside a range will not be
read again soon, so they leave the process's working set first
(`VirtualUnlock` on unlocked pages on Windows, `madvise(MADV_DONTNEED)`
elsewhere); a later read brings them back from the file. `GGUFModel::drop_pages`
applies it to one tensor, and the loader calls it for every tensor no host
reads in place once the model is built ([load](inference-load.md)).
