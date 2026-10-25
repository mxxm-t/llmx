# `src/image/image.hpp` - an image file as the model processor receives it

The front end of image input ([VISION](../VISION.md), section 2): an image file turned into the 8-bit RGB the Qwen vision models' processor receives, and that image sized and resized as the processor resizes it.

`image::decode_image` reads a PNG or a JPEG by its first bytes and refuses GIF, WebP and anything else by name.
It then turns the image by its EXIF orientation as Pillow's `exif_transpose` does, which transformers' `load_image` calls:
- the orientation is IFD0's tag 0x0112, read as Pillow reads it: a directory cut short keeps the entries before the cut, an entry of a type Pillow does not know is skipped, and a tag of several values gives its first;
- with no orientation tag, the first `tiff:Orientation` in the XMP followed by `="` or `>` and a digit, as Pillow falls back to it;
- orientations 2 to 8 mirror, rotate, transpose or transverse the image, and any other value leaves it as it is;
- EXIF that is not TIFF-structured, an orientation of a non-integer type or with no value is refused; Pillow raises on most of these, and where it may still open such a file llmx refuses it rather than give other pixels.

`image::smart_resize` is HF's: the height and width nearest the image's, each a multiple of the factor, holding between the minimum and maximum pixels, its float steps those of the Python reference (a quotient rounded half to even, then scaled by a square root and floored or ceiled), and an aspect ratio past 200 refused.

`image::resize_bicubic` is torchvision's v2 resize of an 8-bit image with bicubic interpolation and antialiasing, as the processor calls it: the width first and then the height, a pass skipped where its size is unchanged, each output sample a fixed-point sum of 16-bit weights of the Keys kernel (a = -0.5) widened by the downscale factor, the weights given the most fraction bits that keep the largest inside 16 bits, and each pass rounded and clamped to 8 bits.

`image-decode` holds all three to the reference processor's bytes over its fixtures (tests/data/image_decode.json).
