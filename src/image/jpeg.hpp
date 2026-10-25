#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "image/raster.hpp"

// JPEG decoded to 8-bit RGB with the bytes libjpeg-turbo gives Pillow: Huffman baseline and progressive frames, 8-bit gray or three components, 4:4:4, 4:2:2 and 4:2:0.
// What it does not read is refused by name (docs/src/image-jpeg.md).

namespace image {

namespace jpeg_detail {

inline constexpr uint8_t kZigzag[80] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    // Past the block a corrupt run lands on the last coefficient, as libjpeg's table extends.
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

struct Huffman {
    bool defined = false;
    int32_t maxcode[18] = {}, valptr[17] = {}, mincode[17] = {};
    uint8_t values[256] = {};

    void build(const uint8_t* counts, const uint8_t* vals, int total) {
        std::memcpy(values, vals, size_t(total));
        int code = 0, k = 0;
        for (int len = 1; len <= 16; ++len) {
            valptr[len] = k;
            mincode[len] = code;
            code += counts[len - 1];
            k += counts[len - 1];
            maxcode[len] = counts[len - 1] ? code - 1 : -1;
            if (code > (1 << len)) throw std::runtime_error("jpeg: a Huffman table is oversubscribed");
            code <<= 1;
        }
        maxcode[17] = 0x7fffffff;
        defined = true;
    }
};

// The entropy-coded bytes of a scan: 0xFF 0x00 is a data byte, and any other marker ends the data; the bits a decode needs past it are refused.
struct BitReader {
    const uint8_t* p;
    size_t n, at;
    uint32_t buf = 0;
    int count = 0;
    bool marker = false;

    void fill() {
        while (count <= 24) {
            if (marker || at >= n) return;
            uint8_t b = p[at];
            if (b == 0xff) {
                if (at + 1 >= n) { marker = true; return; }
                const uint8_t next = p[at + 1];
                if (next == 0x00) {
                    at += 2;
                } else if (next == 0xff) {
                    ++at;  // fill bytes before a marker
                    continue;
                } else {
                    marker = true;
                    return;
                }
            } else {
                ++at;
            }
            buf |= uint32_t(b) << (24 - count);
            count += 8;
        }
    }
    int bit() {
        if (count == 0) {
            fill();
            if (count == 0) throw std::runtime_error("jpeg: the entropy-coded data ends before the scan does");
        }
        const int v = int(buf >> 31);
        buf <<= 1;
        --count;
        return v;
    }
    int bits(int k) {
        int v = 0;
        for (int i = 0; i < k; ++i) v = (v << 1) | bit();
        return v;
    }
    int decode(const Huffman& h) {
        if (!h.defined) throw std::runtime_error("jpeg: a scan uses an undefined Huffman table");
        int code = bit(), len = 1;
        while (code > h.maxcode[len]) {
            code = (code << 1) | bit();
            if (++len > 16) throw std::runtime_error("jpeg: a Huffman code no table holds");
        }
        return h.values[h.valptr[len] + code - h.mincode[len]];
    }
    static int extend(int v, int s) { return v < (1 << (s - 1)) ? v - (1 << s) + 1 : v; }
    // Drops the bits left of the current byte and steps past the restart marker `want` (0 to 7).
    void restart(int want) {
        buf = 0;
        count = 0;
        marker = false;
        while (at + 1 < n && p[at] == 0xff && p[at + 1] == 0xff) ++at;
        if (at + 1 >= n || p[at] != 0xff || p[at + 1] != 0xd0 + want) throw std::runtime_error("jpeg: a restart marker is missing or out of order");
        at += 2;
    }
};

struct Component {
    int id = 0, h = 1, v = 1, tq = 0;
    int dc_table = 0, ac_table = 0;
    size_t bw = 0, bh = 0;                // blocks stored, padded to whole MCUs
    size_t width = 0, height = 0;         // samples before upsampling (libjpeg's downsampled size)
    std::vector<int16_t> coefs;           // 64 a block, zigzag undone
    int dc_pred = 0;
    int16_t* block(size_t bx, size_t by) { return coefs.data() + (by * bw + bx) * 64; }
};

// The islow inverse DCT of one block into 8 rows of `out`, `stride` apart, as libjpeg's jidctint computes it, the range limit included.
inline void idct_islow(const int16_t* in, const uint16_t* q, uint8_t* out, size_t stride) {
    constexpr int CB = 13, P1 = 2;
    constexpr int32_t F0298 = 2446, F0390 = 3196, F0541 = 4433, F0765 = 6270, F0899 = 7373, F1175 = 9633, F1501 = 12299,
                      F1847 = 15137, F1961 = 16069, F2053 = 16819, F2562 = 20995, F3072 = 25172;
    int32_t ws[64];
    for (int c = 0; c < 8; ++c) {
        const int16_t* col = in + c;
        if (!col[8] && !col[16] && !col[24] && !col[32] && !col[40] && !col[48] && !col[56]) {
            const int32_t dc = int32_t(col[0]) * q[c] * (1 << P1);
            for (int r = 0; r < 8; ++r) ws[r * 8 + c] = dc;
            continue;
        }
        int32_t z2 = int32_t(col[16]) * q[16 + c], z3 = int32_t(col[48]) * q[48 + c];
        int32_t z1 = (z2 + z3) * F0541;
        int32_t t2 = z1 + z3 * -F1847, t3 = z1 + z2 * F0765;
        z2 = int32_t(col[0]) * q[c];
        z3 = int32_t(col[32]) * q[32 + c];
        int32_t t0 = (z2 + z3) * (1 << CB), t1 = (z2 - z3) * (1 << CB);
        const int32_t t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2;
        t0 = int32_t(col[56]) * q[56 + c];
        t1 = int32_t(col[40]) * q[40 + c];
        t2 = int32_t(col[24]) * q[24 + c];
        t3 = int32_t(col[8]) * q[8 + c];
        z1 = t0 + t3;
        z2 = t1 + t2;
        z3 = t0 + t2;
        int32_t z4 = t1 + t3;
        const int32_t z5 = (z3 + z4) * F1175;
        t0 *= F0298; t1 *= F2053; t2 *= F3072; t3 *= F1501;
        z1 *= -F0899; z2 *= -F2562; z3 *= -F1961; z4 *= -F0390;
        z3 += z5; z4 += z5;
        t0 += z1 + z3; t1 += z2 + z4; t2 += z2 + z3; t3 += z1 + z4;
        constexpr int s = CB - P1;
        constexpr int32_t h = 1 << (s - 1);
        ws[0 * 8 + c] = (t10 + t3 + h) >> s;
        ws[7 * 8 + c] = (t10 - t3 + h) >> s;
        ws[1 * 8 + c] = (t11 + t2 + h) >> s;
        ws[6 * 8 + c] = (t11 - t2 + h) >> s;
        ws[2 * 8 + c] = (t12 + t1 + h) >> s;
        ws[5 * 8 + c] = (t12 - t1 + h) >> s;
        ws[3 * 8 + c] = (t13 + t0 + h) >> s;
        ws[4 * 8 + c] = (t13 - t0 + h) >> s;
    }
    // libjpeg's post-IDCT range limit: the descaled value plus 128, saturated over [-512, 511] and wrapping past it.
    const auto limit = [](int32_t x) -> uint8_t {
        const int32_t i = x & 1023;
        return uint8_t(i < 128 ? i + 128 : i < 512 ? 255 : i < 896 ? 0 : i - 896);
    };
    for (int r = 0; r < 8; ++r) {
        const int32_t* w = ws + r * 8;
        uint8_t* o = out + size_t(r) * stride;
        constexpr int s = CB + P1 + 3;
        constexpr int32_t h = 1 << (s - 1);
        if (!w[1] && !w[2] && !w[3] && !w[4] && !w[5] && !w[6] && !w[7]) {
            const uint8_t v = limit((w[0] + (1 << (P1 + 2))) >> (P1 + 3));
            for (int c = 0; c < 8; ++c) o[c] = v;
            continue;
        }
        int32_t z2 = w[2], z3 = w[6];
        int32_t z1 = (z2 + z3) * F0541;
        int32_t t2 = z1 + z3 * -F1847, t3 = z1 + z2 * F0765;
        int32_t t0 = (w[0] + w[4]) * (1 << CB), t1 = (w[0] - w[4]) * (1 << CB);
        const int32_t t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2;
        t0 = w[7]; t1 = w[5]; t2 = w[3]; t3 = w[1];
        z1 = t0 + t3; z2 = t1 + t2; z3 = t0 + t2;
        int32_t z4 = t1 + t3;
        const int32_t z5 = (z3 + z4) * F1175;
        t0 *= F0298; t1 *= F2053; t2 *= F3072; t3 *= F1501;
        z1 *= -F0899; z2 *= -F2562; z3 *= -F1961; z4 *= -F0390;
        z3 += z5; z4 += z5;
        t0 += z1 + z3; t1 += z2 + z4; t2 += z2 + z3; t3 += z1 + z4;
        o[0] = limit((t10 + t3 + h) >> s);
        o[7] = limit((t10 - t3 + h) >> s);
        o[1] = limit((t11 + t2 + h) >> s);
        o[6] = limit((t11 - t2 + h) >> s);
        o[2] = limit((t12 + t1 + h) >> s);
        o[5] = limit((t12 - t1 + h) >> s);
        o[3] = limit((t13 + t0 + h) >> s);
        o[4] = limit((t13 - t0 + h) >> s);
    }
}

}  // namespace jpeg_detail

// What a JPEG holds apart from its pixels that the conversion reads: its EXIF (the first Exif APP1 with later ones' bodies appended, as Pillow keeps it) and its last XMP APP1, each empty when absent.
struct JpegExtras {
    std::vector<uint8_t> exif;
    std::string xmp;
};

inline bool is_jpeg(const uint8_t* p, size_t n) { return n >= 3 && p[0] == 0xff && p[1] == 0xd8 && p[2] == 0xff; }

inline Raster decode_jpeg(const uint8_t* p, size_t n, JpegExtras* extras = nullptr, uint64_t max_pixels = kMaxDecodedPixels) {
    using namespace jpeg_detail;
    if (!is_jpeg(p, n)) throw std::runtime_error("jpeg: not a JPEG file");
    uint16_t qt[4][64] = {};
    bool qt_defined[4] = {};
    Huffman dc[4], ac[4];
    std::vector<Component> comps;
    size_t width = 0, height = 0;
    int hmax = 1, vmax = 1;
    bool progressive = false, frame = false, saw_jfif = false, saw_adobe = false, ended = false;
    int adobe_transform = -1;
    size_t restart_interval = 0, mcux = 0, mcuy = 0;
    size_t at = 2;

    const auto be16 = [&](size_t i) -> size_t {
        if (i + 2 > n) throw std::runtime_error("jpeg: the file ends inside a marker segment");
        return size_t(p[i]) << 8 | p[i + 1];
    };

    while (!ended) {
        while (at < n && p[at] != 0xff) ++at;  // tolerate stray bytes between segments, as libjpeg does with a warning
        while (at < n && p[at] == 0xff) ++at;
        if (at >= n) throw std::runtime_error("jpeg: the file ends before its end-of-image marker");
        const uint8_t m = p[at++];
        if (m == 0xd9) { ended = true; break; }
        if (m >= 0xd0 && m <= 0xd7) continue;
        if (m == 0x01) continue;
        const size_t len = be16(at);
        if (len < 2 || at + len > n) throw std::runtime_error("jpeg: a marker segment runs past the end of the file");
        const uint8_t* s = p + at + 2;
        const size_t sl = len - 2;
        at += len;
        if (m == 0xdb) {
            for (size_t i = 0; i < sl;) {
                const int pq = s[i] >> 4, tq = s[i] & 15;
                if (tq > 3 || pq > 1) throw std::runtime_error("jpeg: a malformed quantization table");
                const size_t need = 1 + 64 * size_t(pq + 1);
                if (i + need > sl) throw std::runtime_error("jpeg: a quantization table runs past its segment");
                for (int k = 0; k < 64; ++k)
                    qt[tq][kZigzag[k]] = pq ? uint16_t(s[i + 1 + 2 * k] << 8 | s[i + 2 + 2 * k]) : s[i + 1 + k];
                qt_defined[tq] = true;
                i += need;
            }
        } else if (m == 0xc4) {
            for (size_t i = 0; i < sl;) {
                if (i + 17 > sl) throw std::runtime_error("jpeg: a Huffman table runs past its segment");
                const int tc = s[i] >> 4, th = s[i] & 15;
                if (tc > 1 || th > 3) throw std::runtime_error("jpeg: a malformed Huffman table");
                int total = 0;
                for (int k = 0; k < 16; ++k) total += s[i + 1 + k];
                if (total > 256 || i + 17 + size_t(total) > sl) throw std::runtime_error("jpeg: a Huffman table runs past its segment");
                (tc ? ac : dc)[th].build(s + i + 1, s + i + 17, total);
                i += 17 + size_t(total);
            }
        } else if (m == 0xdd) {
            if (sl < 2) throw std::runtime_error("jpeg: a malformed restart interval");
            restart_interval = size_t(s[0]) << 8 | s[1];
        } else if (m == 0xe0) {
            if (sl >= 5 && std::memcmp(s, "JFIF\0", 5) == 0) saw_jfif = true;
        } else if (m == 0xe1) {
            if (extras && sl >= 6 && std::memcmp(s, "Exif\0\0", 6) == 0) {
                if (extras->exif.empty()) extras->exif.assign(s, s + sl);
                else extras->exif.insert(extras->exif.end(), s + 6, s + sl);
            } else if (extras && sl >= 29 && std::memcmp(s, "http://ns.adobe.com/xap/1.0/\0", 29) == 0) {
                extras->xmp.assign(reinterpret_cast<const char*>(s + 29), sl - 29);
            }
        } else if (m == 0xee) {
            if (sl >= 12 && std::memcmp(s, "Adobe", 5) == 0) {
                saw_adobe = true;
                adobe_transform = s[11];
            }
        } else if (m == 0xc0 || m == 0xc1 || m == 0xc2) {
            if (frame) throw std::runtime_error("jpeg: more than one frame");
            if (sl < 6) throw std::runtime_error("jpeg: a malformed frame header");
            if (s[0] != 8) throw std::runtime_error("jpeg: only 8-bit samples are supported");
            height = size_t(s[1]) << 8 | s[2];
            width = size_t(s[3]) << 8 | s[4];
            const int nf = s[5];
            if (height == 0) throw std::runtime_error("jpeg: a frame whose height comes in a DNL marker is not supported");
            if (width == 0) throw std::runtime_error("jpeg: an invalid image size");
            if (nf == 4) throw std::runtime_error("jpeg: CMYK and YCCK images are not supported");
            if (nf != 1 && nf != 3) throw std::runtime_error("jpeg: only gray and three-component images are supported");
            if (sl < 6 + 3 * size_t(nf)) throw std::runtime_error("jpeg: a malformed frame header");
            if (uint64_t(width) * height > max_pixels) throw std::runtime_error("jpeg: the image has more pixels than the limit");
            comps.resize(size_t(nf));
            for (int i = 0; i < nf; ++i) {
                Component& c = comps[size_t(i)];
                c.id = s[6 + 3 * i];
                c.h = s[7 + 3 * i] >> 4;
                c.v = s[7 + 3 * i] & 15;
                c.tq = s[8 + 3 * i];
                if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) throw std::runtime_error("jpeg: a malformed component");
                hmax = std::max(hmax, c.h);
                vmax = std::max(vmax, c.v);
            }
            for (Component& c : comps) {
                const int rh = hmax / c.h, rv = vmax / c.v;
                if (hmax % c.h || vmax % c.v || rh > 2 || rv > 2 || (rh == 1 && rv == 2))
                    throw std::runtime_error("jpeg: only 4:4:4, 4:2:2 and 4:2:0 sampling are supported");
            }
            if (nf == 1) hmax = vmax = comps[0].h = comps[0].v = 1;  // a lone component is never interleaved
            mcux = (width + 8 * size_t(hmax) - 1) / (8 * size_t(hmax));
            mcuy = (height + 8 * size_t(vmax) - 1) / (8 * size_t(vmax));
            for (Component& c : comps) {
                c.width = (width * size_t(c.h) + size_t(hmax) - 1) / size_t(hmax);
                c.height = (height * size_t(c.v) + size_t(vmax) - 1) / size_t(vmax);
                c.bw = mcux * size_t(c.h);
                c.bh = mcuy * size_t(c.v);
                c.coefs.assign(c.bw * c.bh * 64, 0);
            }
            progressive = m == 0xc2;
            frame = true;
        } else if ((m >= 0xc3 && m <= 0xcf) && m != 0xc4 && m != 0xc8 && m != 0xcc) {
            throw std::runtime_error(m >= 0xc9 ? "jpeg: arithmetic coding is not supported" : "jpeg: lossless and hierarchical JPEG are not supported");
        } else if (m == 0xcc) {
            throw std::runtime_error("jpeg: arithmetic coding is not supported");
        } else if (m == 0xda) {
            if (!frame) throw std::runtime_error("jpeg: a scan before the frame header");
            if (sl < 1) throw std::runtime_error("jpeg: a malformed scan header");
            const int ns = s[0];
            if (ns < 1 || ns > 4 || sl < 4 + 2 * size_t(ns)) throw std::runtime_error("jpeg: a malformed scan header");
            std::vector<Component*> sc;
            for (int i = 0; i < ns; ++i) {
                const int id = s[1 + 2 * i];
                Component* c = nullptr;
                for (Component& x : comps) if (x.id == id) c = &x;
                if (!c) throw std::runtime_error("jpeg: a scan names a component the frame lacks");
                c->dc_table = s[2 + 2 * i] >> 4;
                c->ac_table = s[2 + 2 * i] & 15;
                if (c->dc_table > 3 || c->ac_table > 3) throw std::runtime_error("jpeg: a malformed scan header");
                sc.push_back(c);
            }
            const int ss = s[1 + 2 * ns], se = s[2 + 2 * ns], ah = s[3 + 2 * ns] >> 4, al = s[3 + 2 * ns] & 15;
            if (progressive) {
                if (ss > se || se > 63 || (ss == 0 && se != 0) || (ss > 0 && ns != 1) || al > 13 || (ah && ah != al + 1))
                    throw std::runtime_error("jpeg: an invalid progressive scan");
            } else if (ss != 0 || se != 63 || ah || al) {
                throw std::runtime_error("jpeg: an invalid baseline scan");
            }
            BitReader br{p, n, at};
            int eobrun = 0;
            for (Component* c : sc) c->dc_pred = 0;

            const auto decode_block = [&](Component& c, int16_t* blk) {
                if (!progressive) {
                    const int t = br.decode(dc[c.dc_table]);
                    if (t > 16) throw std::runtime_error("jpeg: an invalid DC code");
                    c.dc_pred += t ? BitReader::extend(br.bits(t), t) : 0;
                    blk[0] = int16_t(c.dc_pred);
                    for (int k = 1; k < 64;) {
                        const int rs = br.decode(ac[c.ac_table]), r = rs >> 4, sz = rs & 15;
                        if (sz) {
                            k += r;
                            blk[kZigzag[std::min(k, 79)]] = int16_t(BitReader::extend(br.bits(sz), sz));
                            ++k;
                        } else if (r == 15) {
                            k += 16;
                        } else {
                            break;
                        }
                    }
                    return;
                }
                if (ss == 0) {
                    if (ah == 0) {
                        const int t = br.decode(dc[c.dc_table]);
                        if (t > 16) throw std::runtime_error("jpeg: an invalid DC code");
                        c.dc_pred += t ? BitReader::extend(br.bits(t), t) : 0;
                        blk[0] = int16_t(uint32_t(c.dc_pred) << al);
                    } else if (br.bit()) {
                        blk[0] = int16_t(blk[0] | (1 << al));
                    }
                    return;
                }
                if (ah == 0) {
                    if (eobrun > 0) { --eobrun; return; }
                    for (int k = ss; k <= se;) {
                        const int rs = br.decode(ac[c.ac_table]), r = rs >> 4, sz = rs & 15;
                        if (sz) {
                            k += r;
                            blk[kZigzag[std::min(k, 79)]] = int16_t(uint32_t(BitReader::extend(br.bits(sz), sz)) << al);
                            ++k;
                        } else if (r < 15) {
                            eobrun = (1 << r) - 1;
                            if (r) eobrun += br.bits(r);
                            break;
                        } else {
                            k += 16;
                        }
                    }
                    return;
                }
                const int p1 = 1 << al, m1 = -(1 << al);
                int k = ss;
                const auto refine = [&](int16_t& coef) {
                    if (br.bit() && (coef & p1) == 0) coef = int16_t(coef >= 0 ? coef + p1 : coef + m1);
                };
                if (eobrun == 0) {
                    for (; k <= se; ++k) {
                        const int rs = br.decode(ac[c.ac_table]);
                        int r = rs >> 4, sz = rs & 15, val = 0;
                        if (sz) {
                            if (sz != 1) throw std::runtime_error("jpeg: an invalid refinement code");
                            val = br.bit() ? p1 : m1;
                        } else if (r != 15) {
                            eobrun = 1 << r;
                            if (r) eobrun += br.bits(r);
                            break;
                        }
                        do {
                            int16_t& coef = blk[kZigzag[std::min(k, 79)]];
                            if (coef != 0) refine(coef);
                            else if (--r < 0) break;
                            ++k;
                        } while (k <= se);
                        if (val) blk[kZigzag[std::min(k, 79)]] = int16_t(val);
                    }
                }
                if (eobrun > 0) {
                    for (; k <= se; ++k) {
                        int16_t& coef = blk[kZigzag[k]];
                        if (coef != 0) refine(coef);
                    }
                    --eobrun;
                }
            };

            size_t units, per_row;
            if (ns == 1) {
                per_row = (sc[0]->width + 7) / 8;
                units = per_row * ((sc[0]->height + 7) / 8);
            } else {
                per_row = mcux;
                units = mcux * mcuy;
            }
            int next_rst = 0;
            for (size_t u = 0; u < units; ++u) {
                if (restart_interval && u && u % restart_interval == 0) {
                    br.restart(next_rst);
                    next_rst = (next_rst + 1) & 7;
                    eobrun = 0;
                    for (Component* c : sc) c->dc_pred = 0;
                }
                const size_t ux = u % per_row, uy = u / per_row;
                if (ns == 1) {
                    decode_block(*sc[0], sc[0]->block(ux, uy));
                } else {
                    for (Component* c : sc)
                        for (int v = 0; v < c->v; ++v)
                            for (int h = 0; h < c->h; ++h)
                                decode_block(*c, c->block(ux * size_t(c->h) + size_t(h), uy * size_t(c->v) + size_t(v)));
                }
            }
            // Past the scan's data: give back what the reader buffered and resume at the next marker.
            at = br.at;
        }
        // Other markers (APPn, COM) are skipped by their length.
    }
    if (!frame) throw std::runtime_error("jpeg: no frame header");
    for (const Component& c : comps)
        if (!qt_defined[c.tq]) throw std::runtime_error("jpeg: a component uses an undefined quantization table");

    // Sample planes, padded to whole blocks.
    std::vector<std::vector<uint8_t>> planes(comps.size());
    for (size_t ci = 0; ci < comps.size(); ++ci) {
        Component& c = comps[ci];
        const size_t stride = c.bw * 8;
        planes[ci].assign(stride * c.bh * 8, 0);
        for (size_t by = 0; by < c.bh; ++by)
            for (size_t bx = 0; bx < c.bw; ++bx)
                idct_islow(c.block(bx, by), qt[c.tq], planes[ci].data() + by * 8 * stride + bx * 8, stride);
        c.coefs.clear();
        c.coefs.shrink_to_fit();
    }

    // Each component at full size, by libjpeg's fancy upsampling where its width passes two samples and by replication otherwise.
    const auto full = [&](size_t ci) -> std::vector<uint8_t> {
        const Component& c = comps[ci];
        const size_t stride = c.bw * 8, cw = c.width, chh = c.height;
        const uint8_t* in = planes[ci].data();
        const int rh = hmax / c.h, rv = vmax / c.v;
        const size_t ow = cw * size_t(rh), oh = chh * size_t(rv);
        std::vector<uint8_t> out(ow * oh);
        if (rh == 1 && rv == 1) {
            for (size_t y = 0; y < oh; ++y) std::memcpy(&out[y * ow], in + y * stride, ow);
            return out;
        }
        const bool fancy = cw > 2;
        if (rv == 1) {
            for (size_t y = 0; y < oh; ++y) {
                const uint8_t* r = in + y * stride;
                uint8_t* o = &out[y * ow];
                if (!fancy) {
                    for (size_t x = 0; x < cw; ++x) o[2 * x] = o[2 * x + 1] = r[x];
                    continue;
                }
                o[0] = r[0];
                o[1] = uint8_t((r[0] * 3 + r[1] + 2) >> 2);
                for (size_t x = 1; x + 1 < cw; ++x) {
                    const int v = r[x] * 3;
                    o[2 * x] = uint8_t((v + r[x - 1] + 1) >> 2);
                    o[2 * x + 1] = uint8_t((v + r[x + 1] + 2) >> 2);
                }
                const size_t l = cw - 1;
                o[2 * l] = uint8_t((r[l] * 3 + r[l - 1] + 1) >> 2);
                o[2 * l + 1] = r[l];
            }
            return out;
        }
        for (size_t y = 0; y < chh; ++y) {
            for (int half = 0; half < 2; ++half) {
                const uint8_t* r0 = in + y * stride;
                const size_t ny = half == 0 ? (y ? y - 1 : 0) : (y + 1 < chh ? y + 1 : chh - 1);
                const uint8_t* r1 = in + ny * stride;
                uint8_t* o = &out[(2 * y + size_t(half)) * ow];
                if (!fancy) {
                    for (size_t x = 0; x < cw; ++x) o[2 * x] = o[2 * x + 1] = r0[x];
                    continue;
                }
                int this_sum = r0[0] * 3 + r1[0], next_sum = r0[1] * 3 + r1[1], last_sum;
                o[0] = uint8_t((this_sum * 4 + 8) >> 4);
                o[1] = uint8_t((this_sum * 3 + next_sum + 7) >> 4);
                last_sum = this_sum;
                this_sum = next_sum;
                for (size_t x = 1; x + 1 < cw; ++x) {
                    next_sum = r0[x + 1] * 3 + r1[x + 1];
                    o[2 * x] = uint8_t((this_sum * 3 + last_sum + 8) >> 4);
                    o[2 * x + 1] = uint8_t((this_sum * 3 + next_sum + 7) >> 4);
                    last_sum = this_sum;
                    this_sum = next_sum;
                }
                const size_t l = cw - 1;
                o[2 * l] = uint8_t((this_sum * 3 + last_sum + 8) >> 4);
                o[2 * l + 1] = uint8_t((this_sum * 4 + 7) >> 4);
            }
        }
        return out;
    };

    Raster img{uint32_t(width), uint32_t(height)};
    if (comps.size() == 1) {
        const std::vector<uint8_t> g = full(0);
        const size_t gw = comps[0].width;
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                uint8_t* px = img.at(x, y);
                px[0] = px[1] = px[2] = g[y * gw + x];
            }
        return img;
    }
    // libjpeg-turbo's color space default: JFIF means YCbCr, then the Adobe transform, then component ids R, G, B.
    bool rgb = false;
    if (!saw_jfif) {
        if (saw_adobe) rgb = adobe_transform == 0;
        else rgb = comps[0].id == 'R' && comps[1].id == 'G' && comps[2].id == 'B';
    }
    std::vector<uint8_t> f[3];
    size_t fw[3];
    for (size_t ci = 0; ci < 3; ++ci) {
        f[ci] = full(ci);
        fw[ci] = comps[ci].width * size_t(hmax / comps[ci].h);
    }
    constexpr int SB = 16;
    constexpr int32_t half = 1 << (SB - 1);
    const auto fix = [](double v) { return int32_t(v * 65536.0 + 0.5); };  // FIX(v) at 16 fraction bits
    int32_t cr_r[256], cb_b[256], cr_g[256], cb_g[256];
    for (int i = 0; i < 256; ++i) {
        const int32_t x = i - 128;
        cr_r[i] = (fix(1.40200) * x + half) >> SB;
        cb_b[i] = (fix(1.77200) * x + half) >> SB;
        cr_g[i] = -fix(0.71414) * x;
        cb_g[i] = -fix(0.34414) * x + half;
    }
    const auto clamp = [](int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); };
    for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width; ++x) {
            const int a = f[0][y * fw[0] + x], b = f[1][y * fw[1] + x], c = f[2][y * fw[2] + x];
            uint8_t* px = img.at(x, y);
            if (rgb) {
                px[0] = uint8_t(a);
                px[1] = uint8_t(b);
                px[2] = uint8_t(c);
            } else {
                px[0] = clamp(a + cr_r[c]);
                px[1] = clamp(a + ((cb_g[b] + cr_g[c]) >> SB));
                px[2] = clamp(a + cb_b[b]);
            }
        }
    return img;
}

}  // namespace image
