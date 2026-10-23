# `src/model/weight_rows.hpp` - bounded immutable weight rows

`infer::WeightRows` reads selected rows without mapping or reading an entire
table. It retains a shared `format::FileReader` and a checked file span, or
reads a memory span. The borrowed memory constructor leaves lifetime with its
caller; the owning overload also accepts `shared_ptr<const void>` and keeps
that owner until the `WeightRows` source is destroyed. This can retain a mapped
file or other shared host allocation without coupling the reader to its type.
Row size comes from the existing quantization storage description; reads
preserve the stored bytes without dequantizing them.

`Scratch` reserves raw row storage, sorting indices and a request remap before a read.
A file source owns its readers, one set a table: `kReadThreads - 1` threads, started at its first read, and for direct reads an aligned window of a granule for each of the `kReadThreads` readers. A table reads for one scratch at a time.
`staging_bytes` is those windows' host bytes, `kReadThreads` granules for a direct source and none otherwise, which the fit counts once for the table (`LookupMemory::io_bytes`). Its capacity limits the number of requested
rows, including duplicates. Physical page allocations round up to whole pages.
Concurrent reads use separate scratch objects. Scratch is compatible only with
the same row layout and source I/O mode and alignment.

`read` is `start` then `finish`.
`start` validates the request before changing output, sorts and deduplicates row IDs, and reads contiguous runs together.
From a file it deals the runs to the table's `kReadThreads` readers on a thread `std::async` starts and returns at once; from memory it copies the rows before it returns.
`finish` waits for the read and returns the rows, rethrowing its failure; a `start` over a read never finished waits for that read and drops it, and a `Scratch` destroyed with a read in flight waits for it. Its result holds unique rows in
ascending ID order and a remap restoring the original request order and
duplicates. Buffered reads land in the row storage directly. Direct reads align their file ranges and use their reader's aligned
window, in chunks of a granule, including when a tensor starts inside a granule or a row exceeds the
window. A rounded read beyond EOF succeeds only if every requested byte exists.

The returned pointers borrow scratch and remain valid only until its next start,
move or destruction. An I/O failure throws without returning a result; it may
have overwritten scratch, so an earlier result must not be consumed after a
failed read. Invalid IDs, incompatible scratch and capacity refusals occur
before row output is touched. An empty request returns an empty result.

The `weight-rows` CTest fixture checks raw Q8_0 rows of width 160, request order,
duplicates, shape and extent refusals, unaligned spans, rows larger than the
read window, concurrent reads, and direct I/O where the filesystem supports it.
It checks a read started ahead against `read`, `finish` without a start refused, a start over an abandoned read, a scratch dropped with its read in flight, and a table of 600 rows read at every third row and some twice, more runs than readers.
It also drops the caller's memory-owner reference, reads successfully while the
source retains the allocation, and checks that Scratch does not extend that
lifetime. On POSIX it truncates an open source and requires a short-read failure, at `finish` for a read started ahead.

## Runtime integration

An explicit `TensorView::row_source` connects this reader to model execution.
The layer declares a mixer gather role and requested rows per token in
`LayerPlan::lookup`; its architecture fills IDs through `lookup_ids`. The
runtime validates the original tensor/source geometry, skips full-table
adoption, and fits bounded packed backend storage, reader scratch and remap
arrays. An external row source requires an unsharded gather role; a tensor
group shares the host read and replicates the requested packed rows.

The runtime's `start_lookup` and `finish_lookup`, used by single-device and tensor-group stages,
start the read once on the group's first member as the stage begins and finish it before the layer, upload the unique packed rows
to each member, and supply shared remapped IDs through `PreparedLookup`. Existing
backend embedding decodes those rows in request order. Normal and apart
execution have independent scratch and packed buffers. Backend writes consume
the host bytes before returning, and backend storage follows the execution
context's ticket lifetime. File-backed GPU stages use the server's recorder
outside its scheduler lock; the read runs on the table's readers while that thread records the layers before it.

File sources retain their reader. Memory sources may borrow their span or
retain a caller-supplied shared owner; Scratch never owns either source. The
ordinary GGUF loader constructs bounded sources from declared tensor spans and
applies the placement residency policy. This reader has
no cross-pass row cache or concurrent-miss sharing, and reads no further ahead than its own stage. Addressing stays in the architecture and
decoding in the backend; immutable table rows use no conversation-cache policy.
