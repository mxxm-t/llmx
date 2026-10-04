# `src/format/file_digest.hpp` - a file's SHA-256, cached by its stamp

A file's SHA-256 and a cache of it, so a large model file is read for its digest once; the disk tier's entries carry the digest as part of their identity (`docs/DISK-TIER.md`, The entry file).

- `file_sha256(path)` reads the whole file in order in 64 MiB pieces through the file cache (`FileReader`) and returns its SHA-256 as lowercase hex (`core::Sha`, with the SHA extensions where the CPU has them: about 1.4 GB/s on an EPYC 7262, so a 27 GB file in about twenty seconds where the disk keeps up).
- `file_stamp(path)` gives what a cached digest is held to (`FileStamp`): the file's size, its modification time in nanoseconds and its inode, on Windows its file index.
- `cached_file_sha256(dir, path)` returns the digest cached in directory `dir` for the file's absolute path where it was written with the file's present stamp, and otherwise reads the file, returns its digest and caches it: a small text file a path, named by the SHA-256 of the path and written whole or not at all (`OutputFile`); a digest whose file's stamp changed while it was read is returned but not cached.
