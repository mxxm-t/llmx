# `src/core/file_sync.hpp` - making a file durable

`core::sync_handle(h)` flushes an open file to the disk with the strongest call its system has: `FlushFileBuffers` on Windows, `F_FULLFSYNC` on macOS with `fsync` standing in where the file system refuses it (`fsync` alone leaves the bytes in the drive's cache there), `fdatasync` elsewhere (`fsync` for a directory).
`core::sync_file(path)` opens a file by its path and does the same, and `core::sync_directory(dir)` flushes a directory so a rename in it reaches the disk too; Windows keeps no directory to flush, so it does nothing there.
`sync_handle` and `sync_file` report whether the flush succeeds. `sync_directory` returns no status and ignores directory-open or flush failures; on Windows it is a no-op.

- `format::FileWriter::sync` and the disk tier's store (`server::DiskStore`: an entry's directory, its identity file, a claimed file and a repaired header) flush only through it, so no caller has a weaker variant.
- `core::sync_calls()` counts the calls; `disk-store` holds a write to a file flush and a directory flush and a claim to the same through it.
- The file is out of the numerics fingerprint (`cmake/numerics-sources.txt`): a flush changes when a file is durable, not the bytes it holds.
