# `src/core/utf8.hpp` - UTF-8 encoding and lengths

The one owner of UTF-8 encoding and validation, in namespace `utf8`.
Three functions, each written once, for the JSON parser, the tokenizer and the server.

- `encode(cp)`: the UTF-8 bytes of a code point at or below U+10FFFF.
  The JSON parser uses it for `\uXXXX` escapes and the tokenizer for its byte-to-character map.
- `lead_length(byte)`: the length a lead byte announces, read from that byte alone.
  ASCII and continuation bytes count 1 and any byte from 0xF0 up counts 4, so a reader that steps by it always moves on, even over invalid input.
  The tokenizer steps through text and tokens with it, and the server's `utf8_complete` uses it to hold back a character split across tokens.
- `valid_length(s, i)`: the length of the valid sequence at `s[i]`, or 0.
  A sequence is valid when its lead byte is ASCII or 0xC2 to 0xF4, its continuation bytes are all present, and its value is not overlong, not a surrogate and not above U+10FFFF.
  Control characters are valid here; refusing them is the JSON parser's own rule.
  The JSON parser refuses a string at the first 0, and the server's `utf8_sanitize` writes U+FFFD for each byte where it is 0.

The two lengths differ on purpose.
The tokenizer has to turn any bytes into ids, so it stays lenient.
JSON carries only characters, so what the parser accepts and what the server writes into JSON meet the same strict rule.

The `json` CTest covers `valid_length` through the parser's accepted and refused strings, the `server-utf8` CTest covers it through `utf8_sanitize` and `lead_length` through `utf8_complete`, and the tokenizer suites cover `encode` and `lead_length`.
