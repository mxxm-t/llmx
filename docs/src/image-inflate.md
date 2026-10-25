# `src/image/inflate.hpp` - zlib inflate

`image::inflate_zlib` inflates a zlib stream (RFC 1950 around RFC 1951's deflate) and checks its Adler-32, for PNG's image data and its compressed text chunks.
Stored, fixed and dynamic Huffman blocks are read by canonical code a bit at a time; a preset dictionary is refused.
Every read is bounds-checked and the output never grows past the size the caller gives, so a malformed stream, one that holds more than the image needs or a distance before its start is refused by name rather than read past its end.
`image-decode` reaches it through every PNG fixture, and the damage sweep of that test through prefixes and flipped bits.
