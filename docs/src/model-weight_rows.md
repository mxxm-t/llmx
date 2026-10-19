# `src/model/weight_rows.hpp` - bounded immutable weight rows

`infer::WeightRows` reads selected rows without mapping or reading an entire
table. It retains a shared `format::FileReader` and a checked file span, or
reads a memory span. The borrowed memory constructor leaves lifetime with its
caller; the owning overload also accepts `shared_ptr<const void>` and keeps
that owner until the `WeightRows` source is destroyed. This can retain a mapped
file or other shared host allocation without coupling the reader to its type.
Row size comes from the existing quantization storage description; reads
preserve the stored bytes without dequantizing them.

`Scratch` reserves raw row storage, sorting indices, a request remap and a
bounded I/O window before a read. Its capacity limits the number of requested
rows, including duplicates. Physical page allocations round up to whole pages.
Concurrent reads use separate scratch objects. Scratch is compatible only with
the same row layout and source I/O mode and alignment.

`read` validates the request before changing output, sorts and deduplicates
row IDs, and reads contiguous runs together. Its result holds unique rows in
ascending ID order and a remap restoring the original request order and
duplicates. Direct reads align their file ranges and use the aligned scratch
window, including when a tensor starts inside a granule or a row exceeds the
window. A rounded read beyond EOF succeeds only if every requested byte exists.

The returned pointers borrow scratch and remain valid only until its next read,
move or destruction. An I/O failure throws without returning a result; it may
have overwritten scratch, so an earlier result must not be consumed after a
failed read. Invalid IDs, incompatible scratch and capacity refusals occur
before row output is touched. An empty request returns an empty result.

The `weight-rows` CTest fixture checks raw Q8_0 rows of width 160, request order,
duplicates, shape and extent refusals, unaligned spans, rows larger than the
read window, concurrent reads, and direct I/O where the filesystem supports it.
It also drops the caller's memory-owner reference, reads successfully while the
source retains the allocation, and checks that Scratch does not extend that
lifetime. On POSIX it truncates an open source and requires a short-read failure.

## Runtime integration

An explicit `TensorView::row_source` connects this reader to model execution.
The layer declares a mixer gather role and requested rows per token in
`LayerPlan::lookup`; its architecture fills IDs through `lookup_ids`. The
runtime validates the original tensor/source geometry, skips full-table
adoption, and fits bounded packed backend storage, reader scratch and remap
arrays. An external row source requires an unsharded gather role; a tensor
group shares the host read and replicates the requested packed rows.

The runtime's `prepare_lookup`, used by single-device and tensor-group stages,
calls `read` once on the group's first member, uploads the unique packed rows
to each member, and supplies shared remapped IDs through `PreparedLookup`. Existing
backend embedding decodes those rows in request order. Normal and apart
execution have independent scratch and packed buffers. Backend writes consume
the host bytes before returning, and backend storage follows the execution
context's ticket lifetime. File-backed GPU stages use the server's recorder
outside its scheduler lock; the read itself is synchronous on that thread.

File sources retain their reader. Memory sources may borrow their span or
retain a caller-supplied shared owner; Scratch never owns either source. The
ordinary GGUF loader constructs bounded sources from declared tensor spans and
applies the placement residency policy. This reader has
no cross-pass row cache, concurrent-miss sharing or lookahead prefetch. Addressing stays in the architecture and
decoding in the backend; immutable table rows use no conversation-cache policy.
