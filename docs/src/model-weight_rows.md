# `src/model/weight_rows.hpp` - bounded immutable weight rows

`infer::WeightRows` reads selected rows from a table's file without mapping or reading the entire
table. It retains a shared `format::FileReader` and a checked file span.
Row size comes from the existing quantization storage description; reads
preserve the stored bytes without dequantizing them.

`Scratch` reserves raw row storage, sorting indices and a request remap before a read.
A file source owns its readers, one set a table: `kReadThreads - 1` threads, started at its first read, and for direct reads an aligned window of a granule for each of the `kReadThreads` readers. A table reads for one scratch at a time.
`host_bytes` is the table's own host bytes, its store and those windows, `kReadThreads` granules for a direct source and none otherwise, which the fit counts once for the table (`LookupMemory::io_bytes`). A scratch's capacity limits the number of requested
rows, including duplicates. Physical page allocations round up to whole pages.
Concurrent reads use separate scratch objects. Scratch is compatible only with
the same row layout.

`hold` gives the table a store of its hot rows within a number of host bytes, before its first read, or none for 0 (docs/CACHE-PLAN.md, section 9); placement gives it its share (`hold_rows` in [place](model-place.md)).
The store packs its rows a slot each in an arena of host pages committed as slots fill, finds them through an open-addressing index of {row id, slot} at load 3/4 with linear probing and backward-shift deletion, and gives up a slot by CLOCK, one reference byte a slot.
`store_bytes_for` is what a store of so many slots takes in all, each part in whole pages, `store_slots_within` the most slots a size holds, and `store_bytes` the table's store.

A read is `start` then `finish`, which a stage calls apart, across the layers before the one that uses the rows.
`start` validates the request before changing output, sorts and deduplicates row IDs, and reads contiguous runs together.
It returns at once, the read running on a thread `std::async` starts, under the table's readers' mutex for the whole task: a row the store holds is copied from it and never read from the file, the missing runs are dealt to the table's `kReadThreads` readers, and once every read has succeeded the missing rows are put in the store, coming in referenced so a read gives up older rows before its own; a lookup never meets a put, and a failed read puts nothing.
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
It checks a read started ahead against the file's rows, `finish` without a start refused, a start that fails after its checks leaving `finish` refusing, a start over an abandoned read, a scratch dropped with its read in flight, and a table of 600 rows read at every third row and some twice, more runs than readers.
On POSIX it truncates an open source and requires a short-read failure, at `finish` for a read started ahead.
The store's checks: its real bytes and the slots a size holds; a held table's rows; and on POSIX, rewriting the file between reads, a held row read from memory, the store kept to its slots with a read's own rows kept and older ones given up by CLOCK, every row read from the file once it holds none, and nothing published by a read that failed.

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

A source retains its reader; Scratch never owns the source. The
ordinary GGUF loader constructs bounded sources from declared tensor spans, buffered readers in `auto` and `mapped` and direct ones in `direct`, and
applies the placement residency policy. Its store is the table's only cross-pass row cache; it shares no misses between concurrent reads and reads no further ahead than its own stage. Addressing stays in the architecture and
decoding in the backend; immutable table rows use no conversation-cache policy, only the share of the host budget placement gives them.
