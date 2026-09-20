# `src/hub/` - model acquisition

This is a native C++ acquisition path invoked by the CLI, independent of model
execution. It uses core JSON and hashing and an external curl process for HTTPS.
It does not import format, quantization or backend code.

`manifest.hpp` validates Hub metadata and selects a quant suffix or explicit
repository file. It pins the returned commit SHA, distinguishes standalone
files from complete shard sets, rejects ambiguous variants and nonportable paths,
and obtains exact sizes plus LFS SHA256 or Git blob SHA1 identities.

`transport.hpp` launches curl directly with fixed `-q --config -` arguments.
URLs and credentials arrive through stdin. The parent opens the output using
native paths and drains bounded stderr without logging raw curl diagnostics.
Child environment omits token variables and `SSLKEYLOGFILE`; ordinary proxy
and certificate environment remains available. Windows inherits only selected
handles. POSIX marks owned descriptors close-on-exec and protects against SIGPIPE.
Owned children are reaped on success or exception. Curl 8.4+ supplies TLS,
credential stripping on cross-origin redirects and bounded transfers. Full
requests require HTTP 200; ranges require 206, exact Content-Range and byte count.
The internal executable override exists for native fake-child tests, not as a
runtime option. See [curl's documentation](https://curl.se/docs/manpage.html).

`pull.hpp` resolves metadata before downloading immutable URLs. Each pull owns
one temporary directory inside the requested cache. Files of at least 16 MiB
use bounded concurrent ranges, with at least 8 MiB per stream. Futures drain
before captured state or temporary files disappear, including exceptional
startup. Retry operates per stream with fresh output, never appends a failed
response, and only handles transient transport failures. Numeric Retry-After
values above 60 seconds stop the request instead of retrying early. Range parts are copied
in order into a final temporary file while hashing, then atomically published.
The hash check also runs on cache hits. Git files include Git's blob header in
their SHA1 input. Completion returns the first shard after every file is ready.
Progress callbacks run on the caller and currently report phases per file.

Temporary cleanup is best effort after failures. Forced process termination can
leave a private temporary directory. Cache publication is atomic per file, not
transactional across a whole shard set, and does not promise power-loss durability.
Cache paths reject symlinks under the caller-selected root; this is not a
sandbox against a malicious local process changing paths concurrently.

`tests/hub_manifest.cpp` checks selection, invalid metadata/paths and standard
hash vectors. `tests/hub_pull.cpp` uses independently pinned payload hashes and
a fake fetcher to verify overlapping ranges, exact assembly, cache repair,
cleanup, retries and Git blob identity. `tests/hub_transport.cpp` runs native
fake children to exercise process lifetime, binary/Unicode paths, credential
handling, status/range validation and concurrent launches. These tests are
offline; real Hub transfers and platform curl availability need integration
validation. The fixture downloader in `tools/` remains Python test tooling and
does not implement the runtime command.
