# `src/format/file_reader.hpp` - a file read at given offsets

`format::FileReader(path)` opens a file for reading at offsets into the caller's memory, by several threads at once; the loader streams the weights a device copies through it in large reads in file order ([load](inference-load.md)), where a mapping would fault them in a page at a time inside the upload's copy.

- `read(offset, dst, bytes)` reads up to `bytes` from `offset` and returns how many it read, fewer only at the end of the file; a read at or past the end returns 0. Errors throw `std::runtime_error` naming the file.
- `size()` is the file's size when it was opened, and `path()` its UTF-8 path.
- `granule()` is the unit a read starts and ends on for the file system to serve it whole: the larger of the page (`core::page_size`) and the file's block size (`st_blksize`), which on ZFS is the dataset's recordsize; the page on Windows.

On Linux and macOS it opens the file `O_RDONLY | O_CLOEXEC`, advises sequential access where `posix_fadvise` exists, and reads with `pread` in a loop that retries on `EINTR`, so reads at different offsets share the descriptor safely.
On Windows it opens the file from its UTF-8 path with `CreateFileW` (`FILE_SHARE_READ`, `FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED`) and reads with `ReadFile` on an `OVERLAPPED` at the offset, each call with its own event, since the handle's own signal is shared by every read in flight on it; `GetOverlappedResult` waits for it, and `ERROR_HANDLE_EOF` is the end of the file. One call reads at most 1 GiB, and `read` loops.

Reads go through the operating system's file cache, so a model read once is served from it again while the host has room; `tests/file_reader.cpp` (CTest `file-reader`) checks the reads, the short read at the end, empty and tiny files, and concurrent readers.
