# `src/core/json.hpp` — minimal JSON parser

Recursive-descent JSON parser, written from scratch (no libs), in namespace
`jmini`.

- `Value`: a tagged union (`Null/Bool/Number/String/Array/Object`) with accessors
  (`isObject/isArray/isString/isNumber`, `get`, `asString/asNumber/asArray`).
- `Parser` / `parse(str)`: full document parse; throws on malformed input or
  trailing data.
- Handles standard JSON: objects, arrays, strings (with escapes incl. `\uXXXX`),
  numbers, `true/false/null`.

Used by the `quantize` path to read `model.json` tensor descriptions.
