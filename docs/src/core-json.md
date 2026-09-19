# `src/core/json.hpp` - minimal JSON parser

Recursive-descent JSON parser, written from scratch (no libs), in namespace
`jmini`.

- `Value`: a tagged record (`Null/Bool/Number/String/Array/Object`) with accessors
  (`isObject/isArray/isString/isNumber`, `get`, `asString/asNumber/asArray`).
- `Parser` / `parse(str)`: recursive document parsing with trailing-data and
  some syntax checks; malformed inputs are not comprehensively rejected.
- Supports objects, arrays, strings, numbers and `true/false/null`. Number and
  escape validation are incomplete. `\uXXXX` currently narrows to one byte;
  Unicode escape decoding and surrogate pairs are not implemented correctly.

Used by the `quantize` path to read `model.json` tensor descriptions.
