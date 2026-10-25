# `src/image/raster.hpp` - an 8-bit RGB image

`image::Raster` is what the image decoders give and the resize takes: a width, a height and three bytes a pixel, rows top to bottom.
`kMaxDecodedPixels`, 89,478,485, is the most pixels a decoder accepts, Pillow's decompression-bomb warning threshold, so an image the reference opens without complaint is never refused for its size ([VISION](../VISION.md), section 6).
