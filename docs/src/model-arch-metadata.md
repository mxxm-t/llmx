# `src/model/arch/metadata.hpp` - typed metadata reads

The reads of a GGUF file's metadata that every architecture's reader
shares, in namespace `infer::metadata`, so a value's type and range checks
and their refusal texts have one owner. Which keys an architecture reads,
their defaults and its other refusals are its reader's
([qwen3](model-arch-qwen3.md), [qwen35](model-arch-qwen35.md)).

- `integer(file, key, fallback = 0)`: a positive INT32, UINT32, INT64 or
  UINT64 value up to `INT_MAX`. An absent key takes `fallback` and is
  refused when that is zero ("missing metadata"); a signed value that is
  zero or negative ("invalid positive integer"), another type ("invalid
  integer type"), and an unsigned zero or a value past `INT_MAX` ("integer
  outside supported range") are refused.
- `real(file, key, fallback)`: a finite positive F32 or F64 value that is a
  nonzero F32, as a double. An absent key takes `fallback`; another type
  ("invalid floating-point type"), a value that is not finite and positive
  or past the F32 range ("invalid positive float") and one that rounds to an
  F32 zero ("float underflow") are refused.
- `option(file, key, supported)`: a string key that is absent or holds the
  one value the reader supports, refused otherwise ("unsupported
  metadata").
- `choice(file, key, supported, refusal)`: an enumerated UINT32 key that is
  absent or holds the one value the reader supports, refused otherwise with
  the reader's own text, as Qwen3's gating function is.
- `boolean(file, key, fallback)`: a boolean; an absent key takes
  `fallback`, and another type is refused ("invalid boolean type").
- `count(file, key, fallback)`: a count, which may be zero, from any of the
  four integer types up to `INT_MAX`; an absent key takes `fallback`, a
  negative value or one past `INT_MAX` is refused ("integer outside
  supported range") and so is another type ("invalid integer type").
  `count_value` is that check on one value.
- `counts(file, key)`, `booleans(file, key)`: an array of counts, each read
  as `count` reads one, and an array of booleans; an absent key gives an
  empty list, a key that is not an array is refused ("invalid array
  type") and so is an element of another type.
