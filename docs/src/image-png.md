# `src/image/png.hpp` - PNG decoding

`image::decode_png` gives a PNG's pixels as 8-bit RGB with the bytes Pillow gives once transformers' `load_image` has converted them to RGB ([VISION](../VISION.md), section 2):
- gray of 1, 2, 4 and 8 bits, scaled to 0..255 as Pillow's unpackers scale them, and replicated;
- palette images of 1, 2, 4 and 8 bits, expanded through the palette;
- RGB, gray with alpha and RGBA of 8 bits, alpha dropped without compositing;
- tRNS ignored, as Pillow's conversion to RGB ignores it;
- every filter and Adam7 interlacing.

Every chunk's CRC is checked, and IHDR first, PLTE before a palette image's data and IEND are required.
Refused by name: 16-bit samples (Pillow's conversion of them to RGB clips rather than scales), a palette index past the palette, an unknown compression, filter or interlace method, and an image past the pixel limit.

`PngExtras` keeps what the orientation reads, as Pillow's chunk readers keep it: the EXIF of an eXIf chunk or of a tEXt chunk named `exif`, whichever comes last, the text of a `Raw profile type exif` chunk, and the XMP of a chunk named `XML:com.adobe.xmp`, from tEXt, zTXt or iTXt, compressed or not.
A compressed text that fails to inflate reads as empty, as Pillow drops it, and one past Pillow's 1 MiB text limit is refused.
