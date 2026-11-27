# `src/image/png.hpp` - PNG decoding

`image::decode_png` gives a PNG's pixels as 8-bit RGB with the bytes Pillow gives once transformers' `load_image` has converted them to RGB ([VISION](../VISION.md), section 2):
- gray of 1, 2, 4 and 8 bits, scaled to 0..255 as Pillow's unpackers scale them, and replicated;
- palette images of 1, 2, 4 and 8 bits, expanded through the palette;
- RGB, gray with alpha and RGBA of 8 bits;
- every filter and Adam7 interlacing.

Where Pillow's `has_transparency_data` holds, the alpha `convert("RGBA")` would give is kept beside the RGB in `Raster::alpha` for a model that composites it (Inkling, `image::Matte::checker`); the others drop it as `convert("RGB")` does:
- gray with alpha and RGBA take their alpha samples;
- a palette's tRNS gives each entry its alpha, entries past the chunk opaque, an empty chunk all opaque, before or after the image data;
- a gray or RGB tRNS key makes the pixels equal to it transparent, compared as Pillow's `convert_transparent` compares it: the key's low byte against the scaled sample, a 1-bit gray key read as 0 or 255, so a 2- or 4-bit key other than 0 seldom matches and an 8-bit key past 255 matches its low byte;
- a tRNS too short for its gray or RGB key is refused, as Pillow refuses to open it.

Every chunk's CRC is checked, and IHDR first, PLTE before a palette image's data and IEND are required.
Refused by name: 16-bit samples (Pillow's conversion of them to RGB clips rather than scales), a palette index past the palette, an unknown compression, filter or interlace method, and an image past the pixel limit.

`PngExtras` keeps what the orientation reads, as Pillow's chunk readers keep it: the EXIF of an eXIf chunk or of a tEXt chunk named `exif`, whichever comes last, the text of a `Raw profile type exif` chunk, and the XMP of a chunk named `XML:com.adobe.xmp`, from tEXt, zTXt or iTXt, compressed or not.
A compressed text that fails to inflate reads as empty, as Pillow drops it; the compressed texts of those chunks inflate within one budget of 1 MiB a file (`kTextLimit`), and a file past it is refused, as Pillow refuses a text past its 1 MiB limit, since a camera's or an editor's EXIF and XMP are a few tens of KiB and a file past the budget has nothing to read, while skipping its chunks would spend a megabyte of inflate on each.
