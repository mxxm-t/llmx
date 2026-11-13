// gpu-port: the arithmetic port.h's CUDA mappings share with the host (port_prmt, port_dot2_i16), checked against models of the AMD instructions the kernels were written for and of the NVIDIA ones that replace them.
#define LLMX_PORT_HOST_MODEL
#include "backends/gpu/kernels/port.h"

#include <cstdint>
#include <cstdio>
#include <random>

namespace {

unsigned byte_of(uint64_t v, unsigned i) { return (unsigned)(v >> (8 * i)) & 0xffu; }

// v_perm_b32: byte i of the result from selector byte i over hi:lo, lo bytes 0-3.
unsigned amd_perm(unsigned hi, unsigned lo, unsigned sel) {
    const uint64_t v = (uint64_t)hi << 32 | lo;
    unsigned r = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned s = byte_of(sel, i);
        unsigned b;
        if (s < 8) b = byte_of(v, s);
        else if (s < 12) b = (byte_of(v, 2 * (s - 8) + 1) & 0x80u) ? 0xffu : 0u;
        else if (s == 12) b = 0;
        else b = 0xffu;
        r |= b << (8 * i);
    }
    return r;
}

// prmt in its default mode: byte i from nibble i over b:a, a bytes 0-3; nibble bit 3 replicates the selected byte's sign.
unsigned ptx_prmt(unsigned a, unsigned b, unsigned c) {
    const uint64_t v = (uint64_t)b << 32 | a;
    unsigned r = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned n = (c >> (4 * i)) & 0xfu;
        const unsigned x = byte_of(v, n & 7u);
        r |= ((n & 8u) ? ((x & 0x80u) ? 0xffu : 0u) : x) << (8 * i);
    }
    return r;
}

// byte_perm under CUDA, built as port.h builds it from __byte_perm(lo, hi, nibbles).
unsigned cuda_byte_perm(unsigned hi, unsigned lo, unsigned sel) {
    const PortPrmt p = port_prmt(sel);
    return (ptx_prmt(lo, hi, p.nibbles) & ~(p.clear | p.ones)) | p.ones;
}

// v_alignbyte and v_alignbit, and __funnelshift_r(lo, hi, s), which shifts hi:lo right by s mod 32.
unsigned amd_alignbyte(unsigned hi, unsigned lo, unsigned n) { return (unsigned)(((uint64_t)hi << 32 | lo) >> (8 * (n & 3))); }
unsigned amd_alignbit(unsigned hi, unsigned lo, unsigned s) { return (unsigned)(((uint64_t)hi << 32 | lo) >> (s & 31)); }
unsigned funnel_r(unsigned lo, unsigned hi, unsigned s) { return (unsigned)(((uint64_t)hi << 32 | lo) >> (s & 31)); }

// v_dot2_i32_i16 without clamp: the products summed in 64 bits, kept mod 2^32.
int amd_sdot2(int a, int b, int s) {
    const int64_t r = (int64_t)s + (int64_t)(int16_t)a * (int16_t)b + (int64_t)(int16_t)(a >> 16) * (int16_t)(b >> 16);
    return (int)(uint32_t)(uint64_t)r;
}

} // namespace

int main() {
    int failures = 0;
    auto require = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };
    std::mt19937 rng(12345);

    // Every selector byte value in every position, the other three random, over random words and words of all-set and sign-only bytes.
    bool perm = true;
    for (unsigned pos = 0; pos < 4; ++pos)
        for (unsigned s = 0; s < 256; ++s)
            for (int k = 0; k < 64; ++k) {
                unsigned sel = rng();
                sel = (sel & ~(0xffu << (8 * pos))) | (s << (8 * pos));
                const unsigned hi = k == 0 ? 0xffffffffu : k == 1 ? 0x80808080u : (unsigned)rng(), lo = k == 0 ? 0u : k == 1 ? 0x7f7f7f7fu : (unsigned)rng();
                if (amd_perm(hi, lo, sel) != cuda_byte_perm(hi, lo, sel)) perm = false;
            }
    require(perm, "byte_perm: every AMD selector byte translated to prmt gives the same bytes");
    bool calls = true;
    for (unsigned sel : {0x0B090A08u, 0x08010A00u, 0x09030B02u, 0x0C010C00u, 0x0C030C02u, 0x07060504u, 0x03020100u})
        for (int k = 0; k < 4096; ++k) {
            const unsigned hi = rng(), lo = rng();
            if (amd_perm(hi, lo, sel) != cuda_byte_perm(hi, lo, sel)) calls = false;
        }
    require(calls, "byte_perm: the kernels' selectors over 4096 random words each");
    static_assert(port_prmt(0x03020100u).nibbles == 0x3210u && port_prmt(0x0C0C0C0Cu).clear == 0xffffffffu, "constant selectors fold at compile time");

    bool align = true;
    for (int k = 0; k < 1 << 16; ++k) {
        const unsigned hi = rng(), lo = rng(), n = rng();
        if (amd_alignbyte(hi, lo, n) != funnel_r(lo, hi, (n & 3u) * 8u)) align = false;
        if (amd_alignbit(hi, lo, n) != funnel_r(lo, hi, n)) align = false;
    }
    require(align, "byte_align and bit_align as a right funnel shift of hi:lo");

    bool dot = true;
    const int edges[] = {0, 1, -1, 0x7fff, -0x8000, 0x7fff7fff, (int)0x80008000u, (int)0x80007fffu, 0x7fffffff, (int)0x80000000u};
    for (int a : edges)
        for (int b : edges)
            for (int s : edges)
                if (amd_sdot2(a, b, s) != port_dot2_i16(a, b, s)) dot = false;
    for (int k = 0; k < 1 << 16; ++k) {
        const int a = (int)rng(), b = (int)rng(), s = (int)rng();
        if (amd_sdot2(a, b, s) != port_dot2_i16(a, b, s)) dot = false;
    }
    require(dot, "dot2_i16: two signed 16-bit products wrap mod 2^32 as the AMD instruction without clamp");
    return failures ? 1 : 0;
}
