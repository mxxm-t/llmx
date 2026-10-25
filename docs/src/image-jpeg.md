# `src/image/jpeg.hpp` - JPEG decoding

`image::decode_jpeg` gives a JPEG's pixels as 8-bit RGB with the bytes libjpeg-turbo gives Pillow, which the reference model processor decodes through ([VISION](../VISION.md), section 2).
It reads baseline and extended Huffman frames (SOF0, SOF1) and progressive Huffman frames (SOF2) of 8-bit samples, one gray or three color components, 4:4:4, 4:2:2 and 4:2:0 sampling, and restart intervals.

The arithmetic is libjpeg's, each step chosen to give its bytes:
- Huffman decoding per the standard, progressive scans with their end-of-band runs and refinement bits, and a run past the block landing on the last coefficient as libjpeg's extended order table lands it;
- the integer inverse DCT (islow): 13-bit constants, two fraction bits kept between the passes, a column or row with no AC term taken by its shortcut, and the range limit that saturates over [-512, 511] and wraps past it;
- fancy upsampling: 4:2:2 and 4:2:0 chroma by libjpeg's triangle filters over the component's own width and height, its edge rows and columns repeated, and by replication where that width is two samples or fewer;
- YCbCr to RGB through 16-bit fixed-point tables.

The color space is libjpeg-turbo's default: YCbCr when a JFIF marker is present, else the Adobe marker's transform, else RGB when the component ids are R, G and B.
`JpegExtras` keeps the EXIF (the first Exif APP1 with later ones' bodies appended, as Pillow keeps it) and the last XMP APP1.

Refused by name: arithmetic coding, lossless and hierarchical frames, 12-bit samples, CMYK and YCCK, 4:4:0 and other sampling, a height given in a DNL marker, a missing or misordered restart marker, and data that would need bits past a marker or the end of the file; a file without its end-of-image marker is refused too.
