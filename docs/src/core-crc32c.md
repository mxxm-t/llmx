# `src/core/crc32c.hpp` - CRC32C

`core::crc32c(crc, data, n)` continues a CRC32C, the Castagnoli polynomial as iSCSI and ext4 use it, over `n` bytes, starting from 0, with the SSE4.2 `crc32` instruction the AVX2 baseline includes, eight bytes a step, so it costs far less than reading the bytes from a disk.
The disk tier checks its entry files with it (`docs/DISK-TIER.md`, The entry file); `disk-store` holds it to the standard check value of "123456789".
