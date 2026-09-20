# `src/core/sha.hpp` - streaming file hashes

`core::Sha` implements SHA-256 and SHA-1 with bounded block storage. `update`
accepts byte spans; `hex` returns the digest without modifying the accumulated
state. Input lengths are checked against the 64-bit bit-length representation.
Algorithms follow [FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf).

The Hub downloader uses SHA-256 for LFS payload integrity and SHA-1 only for
the existing Git blob identity protocol. The latter includes the Git blob
header before payload bytes. This is not a general authentication/signature API.
Native tests cover empty, short, multi-block and million-byte standard vectors
with different update boundaries and repeated finalization.
