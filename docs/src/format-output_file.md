# `src/format/output_file.hpp` - Checked conversion output

`format::OutputFile` owns one conversion output until publication. GGUF writing
and raw dequantization use it so late stream errors cannot report success or
leave a partial replacement under a final name.

Construction reserves a unique directory beside the destination and opens its
single temporary file with stream exceptions enabled, binary by default and
text for raw JSON to preserve platform line endings. `write` runs the
serializer and explicitly closes the stream, checking buffered completion too.
Only then can `publish` rename the complete file over the destination.
Errors opening, writing, closing or publishing name the destination in UTF-8.
System failures include their category and numeric code, avoiding code-page
path copies from Windows filesystem exceptions. Destruction
closes any remaining stream and removes only the owned temporary file and
its directory; it never removes a final file.

Destinations must be absent or regular files. Directories, symbolic links and
devices are refused. Replacement needs write access to the parent directory and
space for the new file alongside an existing one. It replaces file contents via
a new file, rather than preserving the old file's permissions or hard links.
There is no fsync or power-loss durability guarantee.
A process killed before cleanup can leave a hidden `.llmx-output-*` directory
beside the destination. After confirming that conversion has stopped, its
leftover staging directory can be removed; it is not a completed output.

Each rename is independent. Raw conversion prepares and closes both files
before publishing JSON then binary. A preparation failure preserves both prior
files; a failure publishing the second can leave the first complete replacement
published. This is not a two-file transaction, and callers must treat a reported
failure as a failed conversion, even when a complete first file exists.

`format-output` checks serialization and stream failures, refusal of premature
publication, replacement, a second publication refused after the first succeeds,
and cleanup. `tests/roundtrip.py` drives actual CLI writes with child-only file
size limits on POSIX, including zero limits that reach buffered completion,
and checks an inaccessible second output and aliased raw output paths.
