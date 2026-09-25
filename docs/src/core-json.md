# `src/core/json.hpp` - minimal JSON parser

Recursive-descent JSON parser, written from scratch (no libs), in namespace
`jmini`.

- `Value`: a tagged record (`Null/Bool/Number/String/Array/Object`) with accessors
  (`isObject/isArray/isString/isNumber`, `get`, `asString/asNumber/asArray`).
- `Parser` / `parse(str)`: recursive parsing of one complete JSON value, with
  only JSON whitespace before and after it. Invalid syntax and trailing data
  throw `std::runtime_error`.
- Supports objects, arrays, strings, numbers and contiguous `true/false/null`
  literals. Object members retain their order and duplicates; `get` returns
  the first matching member, preserving the existing API.
- Strings validate raw UTF-8 and decode JSON escapes, including `\uXXXX` and
  paired UTF-16 surrogates, into UTF-8. Unescaped controls, malformed UTF-8,
  invalid escapes and unpaired surrogates are rejected. Requiring Unicode
  scalar values is this parser's policy; the RFC grammar also permits lone
  surrogate escapes.
- Numbers follow JSON's decimal grammar and convert to finite `double` using
  a stream imbued with the classic locale. Overflow and nonzero values that
  underflow to zero are rejected; signed zero is preserved. This is bounded
  floating-point storage, not exact arbitrary-precision integer storage.
- At most 256 nested arrays/objects are accepted. Primitive values do not add
  to that depth. No separate document-size limit is imposed.
- `quote(text)` returns a quoted JSON string, escaping quotation marks,
  backslashes and control bytes. Its input must already be valid UTF-8; it
  preserves other bytes without validating them. This is a string helper,
  not a serializer for `Value` trees.

The syntax reference is [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259.html).
The depth, numeric-range and Unicode policies above define the supported input
subset. Conversion uses the C++17 standard library without depending on newer
libraries' floating-point `from_chars` support.

Used by the `quantize` path to read `model.json` tensor descriptions and by
`dequantize` to quote the source path and tensor names. Parsing does not validate
tensor schemas, dimension ranges, extent arithmetic or binary payloads. The
`quantize` command checks these conversion requirements after parsing, using
the stored double values; it does not recover the exact decimal spelling.
Safetensors and HF tokenizer/config loading remain future work.

The native Hub path also parses model/file metadata and curl response-header
JSON. Those consumers impose their own document-size limits and validate the
fields they consume, including duplicate manifest fields, sizes and identities.
The server (`server/api.hpp`) parses request bodies with it and builds its
JSON replies with `quote`.

Some standard libraries set a range-error flag for representable subnormals,
including values rounded up to minimum normal. The parser accepts that flag
only for a fully consumed finite nonzero result whose absolute magnitude is at or below minimum normal.
Overflow, malformed numbers and nonzero numbers rounded to zero stay errors.
