# Writes a binary file as a comma-separated list of byte values, the body of an unsigned char array the source includes.
# cmake -DIN=file -DOUT=file.inc -P embed.cmake
file(READ ${IN} hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE ${OUT} "${bytes}\n")
