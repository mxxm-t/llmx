#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "matrix_precision.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, 4); return out; }
// Fixed independent format data: ggml 750127a5173ac6c29c2c159a10404bf184c421ab, decoded-byte SHA256 e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88.
constexpr char grid[] =
    "08080808080808082b080808080808081919080808080808082b0808080808082b2b080808080808190819080808080808191908080808082b19190808080808"
    "192b19080808080808082b08080808082b082b080808080819192b0808080808082b2b0808080808190808190808080808190819080808082b19081908080808"
    "192b08190808080808081919080808082b081919080808081919191908080808082b19190808080819082b190808080808192b19080808082b192b1908080808"
    "192b2b19080808080808082b080808082b08082b080808081919082b08080808082b082b080808081908192b080808080819192b0808080808082b2b08080808"
    "19192b2b080808082b2b2b2b08080808190808081908080808190808190808082b19080819080808192b08081908080808081908190808082b08190819080808"
    "1919190819080808082b19081908080819082b081908080808192b081908080808080819190808082b080819190808081919081919080808082b081919080808"
    "190819191908080808191919190808082b19191919080808192b19191908080808082b191908080819192b1919080808082b2b19190808081908082b19080808"
    "0819082b190808080808192b190808082b08192b190808081919192b1908080819082b2b1908080808192b2b19080808080808082b0808082b0808082b080808"
    "191908082b080808082b08082b080808190819082b080808081919082b08080808082b082b0808082b2b2b082b080808190808192b080808081908192b080808"
    "2b1908192b080808192b08192b080808080819192b080808191919192b0808080808082b2b0808081919082b2b0808082b2b082b2b0808080819192b2b080808"
    "2b082b2b2b080808190808080819080808190808081908082b19080808190808192b08080819080808081908081908082b081908081908081919190808190808"
    "082b19080819080819082b080819080808192b08081908082b192b0808190808192b2b080819080808080819081908082b080819081908081919081908190808"
    "082b0819081908082b2b081908190808190819190819080808191919081908082b19191908190808192b19190819080808082b19081908082b082b1908190808"
    "19192b19081908081908082b081908080819082b081908082b19082b08190808192b082b081908080808192b081908081919192b08190808082b192b08190808"
    "19082b2b0819080808192b2b0819080808080808191908082b080808191908081919080819190808082b0808191908082b2b0808191908081908190819190808"
    "08191908191908082b19190819190808192b19081919080808082b081919080819192b0819190808082b2b081919080819080819191908080819081919190808"
    "2b19081919190808192b08191919080808081919191908082b081919191908081919191919190808082b19191919080819082b191919080808192b1919190808"
    "0808082b191908082b08082b191908081919082b19190808082b082b191908081908192b191908080819192b1919080808082b2b19190808190808082b190808"
    "081908082b1908082b1908082b190808192b08082b190808080819082b190808191919082b190808080808192b190808191908192b190808082b08192b190808"
    "190819192b190808081919192b19080808082b192b1908081908082b2b1908080819082b2b1908080808192b2b19080808080808082b08082b080808082b0808"
    "19190808082b0808082b0808082b080819081908082b080808191908082b08082b191908082b0808192b1908082b080808082b08082b080819192b08082b0808"
    "2b2b2b08082b080819080819082b080808190819082b08082b190819082b0808192b0819082b080808081919082b08082b081919082b080819191919082b0808"
    "082b1919082b080819082b19082b080808192b19082b08080808082b082b08081919082b082b08080819192b082b08082b2b2b2b082b080819080808192b0808"
    "08190808192b080808081908192b08082b081908192b080819191908192b0808082b1908192b080819082b08192b080808080819192b080819190819192b0808"
    "082b0819192b080819081919192b080808191919192b080808082b19192b08081908082b192b08080808192b192b0808080808082b2b0808190819082b2b0808"
    "081919082b2b08082b082b082b2b0808082b2b082b2b08082b2b2b082b2b0808080819192b2b0808192b192b2b2b080819080808080819080819080808081908"
    "2b19080808081908192b08080808190808081908080819082b081908080819081919190808081908082b19080808190819082b080808190808192b0808081908"
    "2b192b080808190808080819080819082b080819080819081919081908081908082b081908081908190819190808190808191919080819082b19191908081908"
    "192b19190808190808082b19080819082b082b190808190819192b1908081908082b2b19080819081908082b080819080819082b080819082b19082b08081908"
    "0808192b080819081919192b08081908082b192b0808190819082b2b0808190808192b2b0808190808080808190819082b080808190819081919080819081908"
    "082b0808190819082b2b080819081908190819081908190808191908190819082b19190819081908192b19081908190808082b08190819082b082b0819081908"
    "19192b0819081908082b2b0819081908190808191908190808190819190819082b19081919081908192b08191908190808081919190819082b08191919081908"
    "1919191919081908082b19191908190819082b191908190808192b19190819080808082b190819082b08082b190819081919082b19081908082b082b19081908"
    "1908192b190819080819192b19081908190808082b081908081908082b081908192b08082b081908080819082b081908191919082b08190819082b082b081908"
    "08192b082b081908080808192b081908191908192b081908190819192b081908081919192b0819081908082b2b0819080819082b2b0819080808192b2b081908"
    "08080808081919082b080808081919081919080808191908082b080808191908190819080819190808191908081919082b19190808191908192b190808191908"
    "08082b080819190819192b0808191908082b2b0808191908190808190819190808190819081919082b19081908191908192b0819081919080808191908191908"
    "2b081919081919081919191908191908082b19190819190819082b190819190808192b19081919080808082b081919082b08082b081919081919082b08191908"
    "082b082b081919081908192b081919080819192b0819190808082b2b08191908190808081919190808190808191919082b19080819191908192b080819191908"
    "08081908191919082b081908191919081919190819191908082b19081919190819082b081919190808192b081919190808080819191919082b08081919191908"
    "1919081919191908082b0819191919081908191919191908081919191919190808082b19191919081908082b191919080819082b191919080808192b19191908"
    "080808082b191908191908082b191908082b08082b191908190819082b191908081919082b19190808082b082b191908190808192b191908081908192b191908"
    "080819192b1919080808082b2b1919082b2b2b2b2b19190819080808082b190808190808082b19082b190808082b1908192b0808082b190808081908082b1908"
    "19191908082b1908082b1908082b190819082b08082b190808080819082b19082b080819082b190819190819082b1908082b0819082b190819081919082b1908"
    "08191919082b190808082b19082b19081908082b082b19080819082b082b190808080808192b19082b080808192b190819190808192b1908082b0808192b1908"
    "19081908192b190808191908192b190808082b08192b190819080819192b190808190819192b190808081919192b1908192b2b19192b19082b082b2b192b1908"
    "081908082b2b1908080819082b2b1908080808192b2b19082b1919192b2b19080808080808082b082b08080808082b081919080808082b08082b080808082b08"
    "1908190808082b080819190808082b082b19190808082b08192b190808082b0808082b0808082b0819192b0808082b082b2b2b0808082b081908081908082b08"
    "0819081908082b080808191908082b082b08191908082b081919191908082b0808192b1908082b080808082b08082b082b2b082b08082b080819192b08082b08"
    "2b2b2b2b08082b081908080819082b080819080819082b080808190819082b082b08190819082b081919190819082b0819082b0819082b080808081919082b08"
    "2b08081919082b081919081919082b081908191919082b080819191919082b0808082b1919082b081908082b19082b080819082b19082b080808192b19082b08"
    "080808082b082b082b2b08082b082b082b082b082b082b08082b2b082b082b082b2b2b082b082b08081908192b082b08080819192b082b08082b082b2b082b08"
    "2b2b082b2b082b08082b2b2b2b082b081908080808192b080819080808192b082b19080808192b08192b080808192b080808190808192b081919190808192b08"
    "082b190808192b0819082b0808192b0808192b0808192b080808081908192b082b08081908192b081919081908192b08082b081908192b081908191908192b08"
    "0819191908192b0808082b1908192b081908082b08192b080819082b08192b080808192b08192b080808080819192b081919080819192b08082b080819192b08"
    "1908190819192b080819190819192b0808082b0819192b081908081919192b080819081919192b080808191919192b082b192b1919192b080808082b19192b08"
    "190808082b192b08081908082b192b08080819082b192b08080808192b192b08192b19192b192b0808080808082b2b0819190808082b2b0819081908082b2b08"
    "08191908082b2b0819080819082b2b0808190819082b2b0808081919082b2b082b2b082b082b2b082b2b2b2b082b2b0819080808192b2b0808190808192b2b08"
    "08081908192b2b081919192b192b2b082b2b08082b2b2b082b082b082b2b2b0808192b192b2b2b08082b082b2b2b2b082b2b082b2b2b2b081908080808080819"
    "08190808080808192b19080808080819192b08080808081908081908080808192b081908080808191919190808080819082b1908080808192b2b190808080819"
    "19082b080808081908192b08080808192b192b080808081908080819080808192b080819080808191919081908080819082b0819080808192b2b081908080819"
    "190819190808081908191919080808192b19191908080819192b19190808081908082b19080808192b082b190808081919192b19080808191908082b08080819"
    "0819082b080808190808192b080808191919192b08080819082b192b0808081919082b2b0808081908192b2b0808081908080808190808192b08080819080819"
    "1919080819080819082b080819080819190819081908081908191908190808192b19190819080819192b19081908081908082b08190808192b082b0819080819"
    "19192b0819080819190808191908081908190819190808192b19081919080819192b08191908081908081919190808192b081919190808191919191919080819"
    "082b19191908081919082b191908081908192b19190808190808082b190808192b08082b190808191919082b19080819082b082b190808191908192b19080819"
    "0819192b1908081908082b2b19080819190808082b080819081908082b080819080819082b0808192b0819082b080819191919082b080819082b19082b080819"
    "08192b082b080819080808192b080819191908192b080819082b08192b080819190819192b080819081919192b08081908082b192b0808191908082b2b080819"
    "0819082b2b08081908080808081908192b080808081908191919080808190819082b0808081908192b2b08080819081919081908081908190819190808190819"
    "2b19190808190819192b19080819081908082b08081908192b082b080819081919192b0808190819082b2b080819081919080819081908190819081908190819"
    "2b19081908190819192b08190819081908081919081908192b081919081908191919191908190819082b19190819081919082b190819081908192b1908190819"
    "0808082b081908192b08082b081908191919082b08190819082b082b081908191908192b081908190819192b0819081908082b2b081908191908080819190819"
    "08190808191908192b19080819190819192b08081919081908081908191908192b081908191908191919190819190819082b19081919081919082b0819190819"
    "08192b081919081908080819191908192b080819191908191919081919190819082b0819191908191908191919190819081919191919081908082b1919190819"
    "2b2b2b19191908191908082b191908190819082b191908190808192b19190819080808082b1908192b0808082b190819191908082b190819082b08082b190819"
    "190819082b190819081919082b19081908082b082b190819190808192b190819081908192b190819080819192b1908190808082b2b19081919192b2b2b190819"
    "19080808082b081908190808082b0819192b0808082b081908081908082b08192b081908082b081919191908082b0819082b1908082b081919082b08082b0819"
    "08192b08082b081908080819082b08192b080819082b081919190819082b0819082b0819082b081919081919082b081908191919082b081908082b19082b0819"
    "0819082b082b08190808192b082b081908080808192b08192b080808192b081919190808192b0819082b0808192b081919081908192b081908191908192b0819"
    "08082b08192b081919080819192b081908190819192b081908081919192b08190808082b192b08192b19192b192b0819190808082b2b0819081908082b2b0819"
    "080819082b2b0819080808192b2b081908080808080819192b080808080819191919080808081919082b08080808191919081908080819190819190808081919"
    "2b19190808081919192b19080808191908082b08080819192b082b080808191919192b0808081919082b2b080808191919080819080819190819081908081919"
    "2b19081908081919192b08190808191908081919080819192b081919080819191919191908081919082b19190808191919082b190808191908192b1908081919"
    "0808082b080819192b08082b080819191919082b08081919082b082b080819191908192b080819190819192b0808191919080808190819190819080819081919"
    "2b19080819081919192b08081908191908081908190819192b081908190819191919190819081919082b19081908191919082b081908191908192b0819081919"
    "08080819190819192b080819190819191919081919081919082b0819190819191908191919081919081919191908191908082b19190819191908082b19081919"
    "0819082b190819190808192b19081919080808082b081919191908082b081919082b08082b081919190819082b081919081919082b08191908082b082b081919"
    "190808192b081919081908192b081919080819192b081919192b2b192b0819190808082b2b081919190808080819191908190808081919192b19080808191919"
    "192b08080819191908081908081919192b081908081919191919190808191919082b19080819191919082b080819191908192b08081919190808081908191919"
    "2b080819081919191919081908191919082b0819081919191908191908191919081919190819191908082b19081919191908082b081919190819082b08191919"
    "0808192b0819191908080808191919192b080808191919191919080819191919082b0808191919191908190819191919081919081919191908082b0819191919"
    "1908081919191919081908191919191908081919191919190808082b19191919190808082b191919081908082b191919080819082b1919192b192b082b191919"
    "080808192b19191908080808082b19192b080808082b191919190808082b1919082b0808082b191919081908082b191908191908082b191908082b08082b1919"
    "19080819082b191908190819082b191908081919082b19192b2b1919082b19190808082b082b191919080808192b191908190808192b191908081908192b1919"
    "08080819192b1919080808082b2b1919192b19082b2b19191919082b2b2b1919082b2b2b2b2b19191908080808082b190819080808082b192b19080808082b19"
    "0808190808082b192b08190808082b191919190808082b19082b190808082b1919082b0808082b1908192b0808082b190808081908082b191919081908082b19"
    "082b081908082b191908191908082b190819191908082b1908082b1908082b190819082b08082b190808192b08082b190808080819082b192b08080819082b19"
    "1919080819082b19082b080819082b191908190819082b190819190819082b1908082b0819082b191908081919082b190819081919082b190808191919082b19"
    "0808082b19082b19192b192b19082b19081908082b082b19080819082b082b19080808192b082b192b1919192b082b1919082b2b2b082b190808080808192b19"
    "1919080808192b19082b080808192b191908190808192b190819190808192b1908082b0808192b191908081908192b190819081908192b190808191908192b19"
    "0808082b08192b191908080819192b190819080819192b190808190819192b190808081919192b192b2b081919192b19082b2b1919192b192b08192b19192b19"
    "080808082b192b190819192b2b192b1919080808082b2b1908190808082b2b1908081908082b2b1919192b19082b2b19082b192b082b2b1908080808192b2b19"
    "2b2b2b08192b2b192b0808192b2b2b1919082b2b2b2b2b19080808080808082b2b0808080808082b191908080808082b082b08080808082b190819080808082b"
    "081919080808082b192b19080808082b08082b080808082b19192b080808082b190808190808082b081908190808082b080819190808082b2b0819190808082b"
    "191919190808082b082b19190808082b19082b190808082b0808082b0808082b1919082b0808082b1908192b0808082b0819192b0808082b190808081908082b"
    "081908081908082b192b08081908082b080819081908082b2b0819081908082b191919081908082b082b19081908082b19082b081908082b08192b081908082b"
    "080808191908082b2b0808191908082b191908191908082b082b08191908082b190819191908082b081919191908082b1908082b1908082b0819082b1908082b"
    "0808192b1908082b192b2b2b1908082b080808082b08082b191908082b08082b2b2b08082b08082b190819082b08082b081919082b08082b190808192b08082b"
    "081908192b08082b080819192b08082b190808080819082b081908080819082b2b1908080819082b192b08080819082b080819080819082b2b0819080819082b"
    "191919080819082b082b19080819082b19082b080819082b080808190819082b2b0808190819082b191908190819082b082b08190819082b190819190819082b"
    "081919190819082b08082b190819082b1908082b0819082b0819082b0819082b0808192b0819082b080808081919082b2b0808081919082b191908081919082b"
    "082b08081919082b190819081919082b081919081919082b08082b081919082b190808191919082b081908191919082b080819191919082b0808082b1919082b"
    "2b2b082b1919082b190808082b19082b081908082b19082b080819082b19082b192b2b082b19082b080808192b19082b08080808082b082b19190808082b082b"
    "19081908082b082b08191908082b082b19080819082b082b08190819082b082b08081919082b082b2b082b2b082b082b19080808192b082b08190808192b082b"
    "08080819192b082b19192b19192b082b2b082b082b2b082b082b19192b2b082b2b2b19192b2b082b2b08082b2b2b082b2b082b2b2b2b082b190808080808192b"
    "081908080808192b192b08080808192b080819080808192b2b0819080808192b191919080808192b082b19080808192b08192b080808192b080808190808192b"
    "2b0808190808192b191908190808192b082b08190808192b190819190808192b081919190808192b08082b190808192b1908082b0808192b0819082b0808192b"
    "0808192b0808192b080808081908192b191908081908192b190819081908192b081919081908192b190808191908192b081908191908192b080819191908192b"
    "2b2b19191908192b190808082b08192b081908082b08192b080819082b08192b080808192b08192b2b192b2b2b08192b080808080819192b2b0808080819192b"
    "191908080819192b082b08080819192b190819080819192b081919080819192b08082b080819192b190808190819192b081908190819192b080819190819192b"
    "0808082b0819192b2b19192b0819192b190808081919192b081908081919192b080819081919192b080808191919192b082b192b1919192b19082b2b1919192b"
    "080808082b19192b2b1908192b19192b08192b192b19192b19080808082b192b08190808082b192b08081908082b192b2b192b08082b192b08080819082b192b"
    "192b2b2b082b192b08080808192b192b192b0819192b192b2b081919192b192b0808192b2b2b192b0808080808082b2b1919080808082b2b2b2b080808082b2b"
    "0819190808082b2b2b082b0808082b2b2b2b2b0808082b2b1908081908082b2b0819081908082b2b0808191908082b2b2b082b2b08082b2b2b2b2b2b08082b2b"
    "0808081919082b2b19192b1919082b2b2b0808082b082b2b2b2b08082b082b2b2b082b082b082b2b082b2b082b082b2b2b2b2b082b082b2b2b08082b2b082b2b"
    "082b082b2b082b2b2b2b082b2b082b2b082b2b2b2b082b2b1908080808192b2b0819080808192b2b0808190808192b2b0808081908192b2b192b082b08192b2b"
    "08192b2b08192b2b0808080819192b2b192b190819192b2b190819192b192b2b2b2b0808082b2b2b082b2b08082b2b2b2b082b2b082b2b2b08191919192b2b2b"
    "2b19082b192b2b2b082b08082b2b2b2b2b2b08082b2b2b2b08082b082b2b2b2b2b082b082b2b2b2b082b2b082b2b2b2b082b082b2b2b2b2b2b2b2b2b2b2b2b2b";
unsigned digit(char c) { return c <= '9' ? unsigned(c - '0') : unsigned(c - 'a' + 10); }
unsigned magnitude(unsigned index, unsigned lane) {
    const size_t at = 2 * (8 * index + lane);
    return 16 * digit(grid[at]) + digit(grid[at + 1]);
}
double half(uint16_t h) {
    const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
    const double value = exponent ? std::ldexp(double(1024 + fraction), int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return h & 0x8000 ? -value : value;
}
void fixture(uint8_t* packed, float* decoded, uint16_t h, unsigned shift) {
    std::memset(packed, 0, 82);
    packed[0] = uint8_t(h); packed[1] = uint8_t(h >> 8);
    for (unsigned group = 0; group < 16; ++group)
        packed[74 + group / 2] |= uint8_t(((shift + group) % 16) << (4 * (group % 2)));
    for (unsigned vector = 0; vector < 32; ++vector) {
        const unsigned index = (shift + vector) % 1024;
        packed[2 + vector] = uint8_t(index);
        packed[66 + vector / 4] |= uint8_t((index / 256) << (2 * (vector % 4)));
        for (unsigned lane = 0; lane < 8; ++lane) {
            const unsigned i = 8 * vector + lane;
            const bool negative = (shift + i) % 3 == 0;
            if (negative) packed[34 + vector] |= uint8_t(1 << lane);
            const double value = half(h) * double(2 * ((shift + i / 16) % 16) + 1) / 8 * magnitude(index, lane);
            decoded[i] = float(negative ? -value : value);
        }
    }
}
std::vector<float> read(backend::Backend& b, backend::Buffer& buffer, size_t count) {
    std::vector<float> values(count);
    b.read(buffer, 0, values.data(), count * sizeof(float));
    return values;
}
std::vector<std::string> paths(backend::Backend& b) {
    backend::MatrixPaths value; b.swap_matrix_paths(value); return value.take();
}
float bf16_input(float value) {
    uint32_t word = bits(value), top = word >> 16;
    const uint32_t low = word & 65535;
    if (low > 32768 || (low == 32768 && (top & 1))) ++top;
    word = top << 16;
    float rounded; std::memcpy(&rounded, &word, 4); return rounded;
}
bool integer_path(backend::Backend& b, backend::Dtype dtype, size_t extent) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(b);
        return extent == 1 ? profile.lookup_integer_dot : profile.prefer_integer_dot;
    }
#else
    (void)b; (void)dtype; (void)extent;
#endif
    return false;
}
size_t integer_calls = 0;
size_t embeddings(backend::Backend& b) {
    const uint16_t edges[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    constexpr size_t rows = 1034, width = 256;
    std::vector<uint8_t> packed(4 + 82 * rows + 4, 0xa5);
    std::vector<float> decoded(rows * width), initial(rows * width + 2, 12345);
    std::vector<uint32_t> ids(rows);
    for (size_t row = 0; row < rows; ++row) {
        fixture(packed.data() + 4 + row * 82, decoded.data() + row * width,
                row < 1024 ? uint16_t(0x3555) : edges[row - 1024], unsigned(row));
        ids[row] = uint32_t(rows - 1 - row);
    }
    auto weights = b.adopt(packed.data(), packed.size());
    auto output = b.alloc(initial.size() * sizeof(float));
    b.write(*output, 0, initial.data(), initial.size() * sizeof(float));
    b.embed({output.get(), 1}, 22, {weights.get(), 1}, width, rows, ids.data(), rows);
    auto got = read(b, *output, initial.size());
    require(got.front() == initial.front() && got.back() == initial.back(), "embedding output guards");
    for (size_t row = 0; row < rows; ++row)
        for (size_t i = 0; i < width; ++i)
            require(bits(got[1 + row * width + i]) == bits(decoded[ids[row] * width + i]), "embedding bits or zero signs");
    const uint32_t selected[] = {1024, 1025, 1027, 0, 0, uint32_t(rows), UINT32_MAX};
    auto id_buffer = b.adopt(selected, sizeof selected);
    b.write(*output, 0, initial.data(), initial.size() * sizeof(float));
    b.embed_ids({output.get(), 1}, 22, {weights.get(), 1}, width, rows, {id_buffer.get(), 0}, 7);
    got = read(b, *output, initial.size());
    for (size_t i = 0; i < got.size(); ++i) {
        uint32_t want = bits(initial[i]);
        if (i > 0 && i <= 7 * width) {
            const size_t row = (i - 1) / width, lane = (i - 1) % width;
            if (selected[row] < rows) want = bits(decoded[selected[row] * width + lane]);
            else want = 0;
        }
        require(bits(got[i]) == want, "device IDs, invalid IDs or embedding guards");
    }
    const uint32_t invalid = uint32_t(rows);
    bool refused = false;
    try { b.embed({output.get(), 1}, 22, {weights.get(), 1}, width, rows, &invalid, 1); }
    catch (const std::runtime_error&) { refused = true; }
    require(refused, "host embedding accepted invalid ID");
    const auto after_invalid = read(b, *output, initial.size());
    for (size_t i = 0; i < got.size(); ++i)
        require(bits(after_invalid[i]) == bits(got[i]), "invalid host ID changed output");
    std::vector<uint8_t> after(packed.size());
    b.read(*weights, 0, after.data(), after.size());
    require(after == packed, "embedding changed weights");
    return rows * width + 7 * width;
}
size_t products(backend::Backend& b, size_t width, size_t rows, size_t cols, bool one_hot, bool dtype_controls = false) {
    const size_t blocks = width / 256, count = rows * cols;
    std::vector<uint8_t> packed(4 + rows * blocks * 82 + 4, 0xa5);
    std::vector<float> decoded(rows * width), input(cols * width + 2, 0), initial(2 * count + 2, 12345);
    for (size_t j = 0; j < rows * blocks; ++j)
        fixture(packed.data() + 4 + j * 82, decoded.data() + j * 256,
                uint16_t((j % 2 ? 0x8000 : 0) | (0x2400 + j % 1024)), unsigned(j * 37));
    input.front() = input.back() = 12345;
    const float ties[] = {1.00390625f, 1.01171875f, 1.005859375f, -1.00390625f, 1.00048828125f};
    for (size_t col = 0; col < cols; ++col) {
        if (one_hot) input[1 + col * width + (137 * col + 31) % width] = ties[col % 5];
        else for (size_t i = 0; i < width; ++i)
            input[1 + col * width + i] = i % 7 == 0 ? ties[(i + col) % 5] : float(int((17 * i + 7 * col) % 61) - 30) / 32;
    }
    auto rounded = input;
    for (auto& value : rounded) value = bf16_input(value);
    auto reconstructed = input;
    const auto twin = testq::row_activations(std::vector<float>(input.begin() + 1, input.end() - 1));
    std::copy(twin.begin(), twin.end(), reconstructed.begin() + 1);
    auto weights = b.adopt(packed.data(), packed.size());
    auto x = b.adopt(input.data(), input.size() * sizeof(float));
    auto output = b.alloc(initial.size() * sizeof(float));
    std::vector<uint32_t> expert_ids(cols, 0);
    std::vector<float> gains(cols, 0.5f);
    auto id_buffer = b.adopt(expert_ids.data(), cols * sizeof(uint32_t));
    auto gain_buffer = b.adopt(gains.data(), cols * sizeof(float));
    const backend::Backend::Routing routing{{id_buffer.get(), 0}, {gain_buffer.get(), 0}, 1, 1};
    size_t checked = 0;
    for (size_t extent : {size_t(1), size_t(16384)}) {
        const backend::RowRun run{cols, extent};
        const backend::RowRuns runs{&run, 1};
        const int operations = width == 768 && rows == 3 && cols == 17 ? 6 : 1;
        for (int op = 0; op < operations; ++op) {
            std::vector<float> f32_result;
            const std::vector<backend::Dtype> dtypes = dtype_controls
                ? std::vector<backend::Dtype>{backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}
                : std::vector<backend::Dtype>{backend::Dtype::f32};
            for (const auto dtype : dtypes) {
                const bool integer = integer_path(b, dtype, extent);
                const bool additive = op == 3 || op == 5;
                const size_t copies = op == 2 ? 2 : 1;
                std::fill(initial.begin(), initial.end(), additive ? 0.25f : 12345.0f);
                initial.front() = initial.back() = 12345;
                b.write(*output, 0, initial.data(), initial.size() * sizeof(float));
                paths(b);
                const backend::CSlice w{weights.get(), 1}, in{x.get(), 1};
                const backend::Slice out{output.get(), 1};
                switch (op) {
                case 0: b.matmul(22, w, in, out, width, rows, cols, runs, dtype); break;
                case 1: b.matmul_logits(22, w, in, out, width, rows, cols, runs, dtype); break;
                case 2: b.matmul_group({{22, w, out, rows}, {22, w, {output.get(), count + 1}, rows}}, in, width, cols, runs, dtype); break;
                case 3: b.matmul_add(22, w, in, out, width, rows, cols, runs, dtype); break;
                case 4: b.matmul_experts({{22, w, out, rows}}, in, width, cols, routing, runs, dtype); break;
                case 5: b.matmul_experts_add(22, w, in, out, width, rows, cols, routing, runs, dtype); break;
                }
                const auto got = read(b, *output, initial.size());
                require(paths(b) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"}, "requested dtype matrix witness");
                if (integer) ++integer_calls;
                if (dtype == backend::Dtype::f32) f32_result = got;
                if (!integer && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8))
                    for (size_t i = 0; i < got.size(); ++i)
                        require(bits(got[i]) == bits(f32_result[i]), "F16/int8 fallback differs from exact F32-input product");
                const auto& oracle_input = dtype == backend::Dtype::bf16 ? rounded : integer ? reconstructed : input;
                require(got.front() == initial.front(), "product leading guard");
                for (size_t i = copies * count + 1; i < got.size(); ++i)
                    require(bits(got[i]) == bits(initial[i]), "product tail/output guard");
                for (size_t copy = 0; copy < copies; ++copy)
                    for (size_t col = 0; col < cols; ++col)
                        for (size_t row = 0; row < rows; ++row) {
                            const float have = got[1 + copy * count + col * rows + row];
                            const double gain = op == 5 ? 0.5 : 1;
                            if (one_hot && !additive && !integer) {
                                const size_t selected = (137 * col + 31) % width;
                                const float want = float(double(decoded[row * width + selected]) * oracle_input[1 + col * width + selected]);
                                require(bits(have) == bits(want), "one-hot product not exact for selected input precision");
                            } else {
                                double want = 0, magnitude_sum = 0, original = 0, rounding = 0;
                                for (size_t i = 0; i < width; ++i) {
                                    const double term = double(decoded[row * width + i]) * oracle_input[1 + col * width + i];
                                    want += term; magnitude_sum += std::fabs(term);
                                    original += double(decoded[row * width + i]) * input[1 + col * width + i];
                                    rounding += std::fabs(double(decoded[row * width + i]) *
                                                         (double(input[1 + col * width + i]) - oracle_input[1 + col * width + i]));
                                }
                                want = gain * want + (additive ? 0.25 : 0);
                                magnitude_sum = gain * magnitude_sum + (additive ? 0.25 : 0);
                                const double nu = 2 * double(width) * 0x1p-24;
                                const double bound = nu / (1 - nu) * magnitude_sum;
                                require(std::isfinite(have) && std::fabs(double(have) - want) <= bound, "float product exceeds gamma(2n) double oracle");
                                if (integer)
                                    require(std::fabs(double(have) - (gain * original + (additive ? 0.25 : 0))) <= gain * rounding + bound,
                                            "integer product exceeds original-input reconstruction bound");
                            }
                            ++checked;
                        }
            }
        }
    }
    require(read(b, *x, input.size()) == input, "product changed input/guards");
    std::vector<uint8_t> after(packed.size());
    b.read(*weights, 0, after.data(), after.size());
    require(after == packed, "product changed weights/guards");
    return checked;
}
size_t precision_products(backend::Backend& b) {
    // Index zero, scale nibble zero and d=1 give eight * (1/8), exactly one.
    std::vector<uint8_t> packed(82, 0);
    packed[1] = 0x3c;
    const auto w = b.adopt(packed.data(), packed.size());
    size_t values = 0;
    for (float sign : {-1.0f, 1.0f}) {
        std::vector<float> input(256, 0);
        input[0] = 2; input[1] = -2; input[2] = sign / 1024;
        const auto twin = testq::row_activations(input);
        require(twin[2] == sign * (16.0f * (2.0f / 32767.0f)), "independent int16 precision anchor");
        const auto x = b.adopt(input.data(), input.size() * sizeof(float));
        const float initial[] = {19, 0, 19};
        const auto out = b.adopt(initial, sizeof initial);
        for (size_t extent : {size_t(1), size_t(16384)})
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            const bool integer = integer_path(b, dtype, extent);
            const backend::RowRun run{1, extent};
            paths(b);
            b.matmul(22, {w.get(), 0}, {x.get(), 0}, {out.get(), 1}, 256, 1, 1, {&run, 1}, dtype);
            const auto got = read(b, *out, 3);
            require(paths(b) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"}, "precision witness");
            if (integer) ++integer_calls;
            const float want = integer ? twin[2] : input[2];
            require(got.front() == 19 && got.back() == 19 && std::isfinite(got[1]) &&
                    std::fabs(double(got[1]) - want) <= std::fabs(double(want)) * 0x1p-20,
                    "small residual beside cancelling peaks loses int16 precision");
            ++values;
        }
    }
    return values;
}
size_t range_products(backend::Backend& b) {
    const uint16_t halves[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    const float inputs[] = {0, 0x1p-145f, 0x1p-120f, 0x1p-24f, 0x1p-14f, 1, 65504, 0x1p90f};
    size_t values = 0;
    for (uint16_t h : halves) {
        // Index one has magnitude 43 in lane zero; nibble 15 reaches coefficient 43*31=1333.
        std::vector<uint8_t> packed(82, 0);
        packed[0] = uint8_t(h); packed[1] = uint8_t(h >> 8);
        std::fill(packed.begin() + 2, packed.begin() + 34, uint8_t(1));
        std::fill(packed.begin() + 74, packed.end(), uint8_t(255));
        const double weight = half(h) * (43.0 * 31 / 8);
        const auto w = b.adopt(packed.data(), packed.size());
        for (float value : inputs)
        for (float sign : {-1.0f, 1.0f}) {
            std::vector<float> input(258, 0);
            input.front() = input.back() = 19; input[1] = sign * value;
            const auto x = b.adopt(input.data(), input.size() * sizeof(float));
            const float initial[] = {19, 0, 19};
            const auto out = b.adopt(initial, sizeof initial);
            for (size_t extent : {size_t(1), size_t(16384)})
            for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8}) {
                const bool integer = integer_path(b, dtype, extent);
                const backend::RowRun run{1, extent};
                paths(b);
                b.matmul(22, {w.get(), 0}, {x.get(), 1}, {out.get(), 1}, 256, 1, 1, {&run, 1}, dtype);
                const auto got = read(b, *out, 3);
                require(paths(b) == std::vector<std::string>{integer ? "block-int16" : "f32"}, "range witness");
                if (integer) ++integer_calls;
                const double want = weight * input[1];
                // F32 scale rounding plus one subnormal scale unit per integer level; no reciprocal oracle on tiny inputs.
                const double repair = integer ? std::fabs(double(value)) * 0x1p-22 + 32767.0 * std::numeric_limits<float>::denorm_min() : 0;
                const double bound = std::fabs(weight) * repair + std::fabs(want) * 0x1p-21 + 2.0 * std::numeric_limits<float>::denorm_min();
                require(got.front() == 19 && got.back() == 19 && std::isfinite(got[1]) && std::fabs(double(got[1]) - want) <= bound,
                        "finite half, maximal coefficient or activation-scale range control");
                ++values;
            }
            require(read(b, *x, input.size()) == input, "range changed input/guards");
        }
    }
    return values;
}
size_t tall_tile_control(backend::Backend& b) {
    if (b.is_cpu()) return 0;
    // 65*9 workgroups pass the MI50 narrow-row threshold, with output tails on both axes.
    constexpr size_t width = 256, rows = 8193, cols = 513;
    std::vector<uint8_t> packed(rows * 82);
    std::vector<float> decoded(rows * width), input(cols * width, 0), initial(rows * cols + 2, 12345);
    for (size_t r = 0; r < rows; ++r)
        fixture(packed.data() + r * 82, decoded.data() + r * width, uint16_t(0x1800 + r % 1024), unsigned(r * 13));
    for (size_t c = 0; c < cols; ++c) input[c * width + (c * 137 + 31) % width] = 1;
    const auto w = b.adopt(packed.data(), packed.size()), x = b.adopt(input.data(), input.size() * sizeof(float));
    const auto out = b.adopt(initial.data(), initial.size() * sizeof(float));
    const backend::RowRun run{cols, 16384};
    paths(b);
    b.matmul(22, {w.get(), 0}, {x.get(), 0}, {out.get(), 1}, width, rows, cols, {&run, 1}, backend::Dtype::f16);
    const auto got = read(b, *out, initial.size());
    const bool integer = integer_path(b, backend::Dtype::f16, 16384);
    require(paths(b) == std::vector<std::string>{integer ? "block-int16" : "f32"}, "tall tile witness");
    if (integer) ++integer_calls;
    require(got.front() == initial.front() && got.back() == initial.back(), "tall tile output guards");
    for (size_t c = 0; c < cols; ++c)
    for (size_t r = 0; r < rows; ++r) {
        const double want = decoded[r * width + (c * 137 + 31) % width];
        require(std::isfinite(got[1 + c * rows + r]) && std::fabs(double(got[1 + c * rows + r]) - want) <= std::fabs(want) * 0x1p-20,
                "tall tile one-hot/tail control");
    }
    return rows * cols;
}
void checks(backend::Backend& b) {
    require(b.supports_type(22), "IQ2_S unsupported");
    static_assert(sizeof(grid) == 16385, "independent grid byte count");
    require(magnitude(0, 0) == 8 && magnitude(1, 0) == 43 && magnitude(256, 0) == 25, "dictionary anchors");
    require(bf16_input(1.00390625f) == 1.0f && bf16_input(1.01171875f) == 1.015625f &&
            bf16_input(1.005859375f) == 1.0078125f && bf16_input(-1.00390625f) == -1.0f &&
            bf16_input(1.00048828125f) == 1.0f, "independent BF16 rounding/tie anchors");
    const size_t embedded = embeddings(b);
    size_t multiplied = 0;
    for (bool one_hot : {false, true}) {
        multiplied += products(b, 256, 3, 1, one_hot);
        multiplied += products(b, 256, 67, 2, one_hot);
        multiplied += products(b, 768, 3, 17, one_hot, true);
        multiplied += products(b, 768, 67, 65, one_hot);
        multiplied += products(b, 4096, 67, 257, one_hot);
    }
    multiplied += products(b, 256, 3, 257, true);
    multiplied += products(b, 4096, 3, 2, false, true);
    multiplied += precision_products(b);
    multiplied += range_products(b);
    multiplied += tall_tile_control(b);
    std::cout << "IQ2_S: " << embedded << " exact embedding values, " << multiplied
              << " product/control values (F32/BF16, profile-selected block-int16, input reconstruction, paths/guards), "
              << integer_calls << " integer calls [ok]\n";
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        try {
            auto b = backend::make_vulkan_backend(0, true);
            checks(*b);
            const auto profile = backend::vulkan_device_profile(*b);
            bool row = false, tile = false, tall = false;
            for (const auto& entry : backend::vulkan_kernel_times(*b)) {
                std::cout << "IQ2_S dispatch: " << entry.first << '\n';
                row |= entry.first.find("matmul_row_iq2_s_dot") != std::string::npos;
                tile |= entry.first.find("matmul_tile_iq2_s_dot") != std::string::npos;
                tall |= entry.first.find("matmul_tile_iq2_s_dot_tall") != std::string::npos;
            }
            if (backend::vulkan_timed_dispatches(*b)) {
                require(row == profile.lookup_integer_dot, "integer-row dispatch differs from profile");
                require(tile == profile.prefer_integer_dot, "integer-tile dispatch differs from profile");
                require(tall == profile.prefer_integer_dot, "tall integer-tile dispatch was not witnessed");
                require((row || tile) == (integer_calls != 0), "integer modules lack block-int16 witnesses");
            }
            return 0;
        } catch (const backend::VulkanUnavailable& e) { std::cout << "IQ2_S Vulkan unavailable: " << e.what() << '\n'; return 77; }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
#else
        std::cout << "IQ2_S Vulkan backend not built\n"; return 77;
#endif
    }
    if (argc != 1) { std::cerr << "usage: llmx-iq2-s-test [--vulkan]\n"; return 1; }
    try {
        backend::CpuBackend b;
        for (int threads : {1, 3}) { b.set_threads(threads); checks(b); }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
