# Original raw-completion depth experiment (2026-09-24)

This frozen experiment did not satisfy its EOS requirement. Its first
Q8_0 baseline request consumed 19,812 prompt tokens and generated 12,956
more, filling the 32,768-token context. It returned `finish: length` after
6,093.875 seconds. The driver stopped at that assertion; no repeat or
candidate long-context comparison was reached.

The original plan, prompt, failure marker, result checkpoint, response,
command record and server log are preserved byte for byte. `artifacts.json`
records their hashes. The response SHA-256 is
`9a861292db3baf5e09e2ad4b67b044a0eb3f48e7faa54e42ae516b22b6a28952`.
Its recorded generated count agrees with the 12,956 returned token IDs.
The parent directory retains the original `run_correctness.py` driver.
Paths in these artifacts identify the original experiment environment.

This is an incomplete output-equality gate, not a candidate correctness
failure or a passing comparison. The later native-chat experiment has its
own frozen request and result set; it does not alter this outcome. Neither
this duration nor the later correctness durations are performance samples.
