# `src/format/file_writer.hpp` - a new file written at offsets

`format::FileWriter(path, direct)` creates a file that must not exist, readable and writable by its owner alone on POSIX (it takes the directory's access on Windows), and writes it at offsets, through the file cache or, with `direct`, around it (`O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` with write-through on Windows), as `FileReader` reads ([file_reader](format-file_reader.md)); the disk tier writes its entries through it (`docs/DISK-TIER.md`).

- A direct writer takes its alignment from the file system (`granule()`, from `statx` on Linux, the sector sizes on Windows, at least a page); a file system that does not take direct writes throws `DirectUnavailable` and leaves no file, as does a system without them.
- `write(offset, src, bytes)` writes the whole range, a direct write starting and ending on the granule from a page; a failed write, the disk full among them, throws naming the file and the system's reason.
- `sync()` flushes what was written to the disk (`fdatasync`, `F_FULLFSYNC` on macOS, where `fsync` leaves the bytes in the drive's cache, `FlushFileBuffers`) and `close()` closes the file; publishing it, by a rename, is the caller's.
