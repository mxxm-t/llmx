#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// A zlib stream (RFC 1950) inflated (RFC 1951), its Adler-32 checked; PNG's image data is one.
// Every read is bounds-checked and the output never grows past the size the caller expects, so a malformed or hostile stream is refused rather than read past its end or inflated without limit.

namespace image {

namespace inflate_detail {

struct Bits {
    const uint8_t* p;
    size_t n, at = 0;
    uint32_t buf = 0;
    int count = 0;

    uint32_t get(int k) {
        while (count < k) {
            if (at >= n) throw std::runtime_error("inflate: the stream ends inside a block");
            buf |= uint32_t(p[at++]) << count;
            count += 8;
        }
        const uint32_t v = buf & ((1u << k) - 1u);
        buf >>= k;
        count -= k;
        return v;
    }
    void align() {
        buf >>= count & 7;
        count -= count & 7;
    }
};

// A canonical Huffman code by its code lengths, decoded a bit at a time over the counts of each length.
struct Huffman {
    uint16_t counts[16] = {};
    std::vector<uint16_t> symbols;

    void build(const uint8_t* lengths, int n) {
        for (int i = 0; i < 16; ++i) counts[i] = 0;
        for (int i = 0; i < n; ++i) ++counts[lengths[i]];
        counts[0] = 0;
        int left = 1;
        for (int len = 1; len < 16; ++len) {
            left = (left << 1) - counts[len];
            if (left < 0) throw std::runtime_error("inflate: a Huffman code is oversubscribed");
        }
        uint16_t offs[16] = {};
        for (int len = 1; len < 15; ++len) offs[len + 1] = uint16_t(offs[len] + counts[len]);
        symbols.assign(size_t(n), 0);
        for (int i = 0; i < n; ++i)
            if (lengths[i]) symbols[offs[lengths[i]]++] = uint16_t(i);
    }
    int decode(Bits& b) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; ++len) {
            code |= int(b.get(1));
            const int count = counts[len];
            if (code - first < count) return symbols[size_t(index + code - first)];
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        throw std::runtime_error("inflate: a code that no Huffman table holds");
    }
};

inline constexpr uint16_t kLengthBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
inline constexpr uint8_t kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
inline constexpr uint16_t kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
inline constexpr uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

inline void codes(Bits& b, std::vector<uint8_t>& out, size_t limit, const Huffman& lit, const Huffman& dist) {
    for (;;) {
        const int sym = lit.decode(b);
        if (sym < 256) {
            if (out.size() >= limit) throw std::runtime_error("inflate: the stream holds more bytes than the image");
            out.push_back(uint8_t(sym));
            continue;
        }
        if (sym == 256) return;
        const int li = sym - 257;
        if (li >= 29) throw std::runtime_error("inflate: an invalid length code");
        const size_t len = kLengthBase[li] + b.get(kLengthExtra[li]);
        const int di = dist.decode(b);
        if (di >= 30) throw std::runtime_error("inflate: an invalid distance code");
        const size_t d = kDistBase[di] + b.get(kDistExtra[di]);
        if (d > out.size()) throw std::runtime_error("inflate: a distance before the start of the stream");
        if (out.size() + len > limit) throw std::runtime_error("inflate: the stream holds more bytes than the image");
        const size_t from = out.size() - d;
        for (size_t i = 0; i < len; ++i) out.push_back(out[from + i]);
    }
}

}  // namespace inflate_detail

// The bytes of zlib stream `p`, at most `limit` of them, refused with a reason when malformed, longer than `limit` or failing its Adler-32.
inline std::vector<uint8_t> inflate_zlib(const uint8_t* p, size_t n, size_t limit) {
    using namespace inflate_detail;
    if (n < 6) throw std::runtime_error("inflate: the stream is too short for a zlib header");
    const unsigned cmf = p[0], flg = p[1];
    if ((cmf & 15) != 8 || (cmf >> 4) > 7) throw std::runtime_error("inflate: not a deflate stream");
    if ((cmf * 256 + flg) % 31 != 0) throw std::runtime_error("inflate: the zlib header check fails");
    if (flg & 32) throw std::runtime_error("inflate: a preset dictionary is not supported");
    Bits b{p + 2, n - 2};
    std::vector<uint8_t> out;
    out.reserve(limit);
    Huffman fixed_lit, fixed_dist;
    bool fixed_built = false;
    for (bool last = false; !last;) {
        last = b.get(1) != 0;
        const uint32_t type = b.get(2);
        if (type == 0) {
            b.align();
            const uint32_t len = b.get(16), nlen = b.get(16);
            if ((len ^ 0xffffu) != nlen) throw std::runtime_error("inflate: a stored block's length check fails");
            if (b.at + len > b.n) throw std::runtime_error("inflate: the stream ends inside a stored block");
            if (out.size() + len > limit) throw std::runtime_error("inflate: the stream holds more bytes than the image");
            out.insert(out.end(), b.p + b.at, b.p + b.at + len);
            b.at += len;
        } else if (type == 1) {
            if (!fixed_built) {
                uint8_t l[288];
                for (int i = 0; i < 144; ++i) l[i] = 8;
                for (int i = 144; i < 256; ++i) l[i] = 9;
                for (int i = 256; i < 280; ++i) l[i] = 7;
                for (int i = 280; i < 288; ++i) l[i] = 8;
                fixed_lit.build(l, 288);
                uint8_t d[30];
                for (auto& x : d) x = 5;
                fixed_dist.build(d, 30);
                fixed_built = true;
            }
            codes(b, out, limit, fixed_lit, fixed_dist);
        } else if (type == 2) {
            const int hlit = int(b.get(5)) + 257, hdist = int(b.get(5)) + 1, hclen = int(b.get(4)) + 4;
            if (hlit > 286 || hdist > 30) throw std::runtime_error("inflate: too many codes in a dynamic block");
            static constexpr uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            uint8_t cl[19] = {};
            for (int i = 0; i < hclen; ++i) cl[order[i]] = uint8_t(b.get(3));
            Huffman lens;
            lens.build(cl, 19);
            uint8_t l[316] = {};
            for (int i = 0; i < hlit + hdist;) {
                const int sym = lens.decode(b);
                int rep = 0;
                uint8_t v = 0;
                if (sym < 16) {
                    l[i++] = uint8_t(sym);
                    continue;
                } else if (sym == 16) {
                    if (i == 0) throw std::runtime_error("inflate: a repeat with no length before it");
                    v = l[i - 1];
                    rep = 3 + int(b.get(2));
                } else if (sym == 17) {
                    rep = 3 + int(b.get(3));
                } else {
                    rep = 11 + int(b.get(7));
                }
                if (i + rep > hlit + hdist) throw std::runtime_error("inflate: code lengths past the table");
                while (rep--) l[i++] = v;
            }
            if (l[256] == 0) throw std::runtime_error("inflate: a dynamic block without an end code");
            Huffman lit, dist;
            lit.build(l, hlit);
            dist.build(l + hlit, hdist);
            codes(b, out, limit, lit, dist);
        } else {
            throw std::runtime_error("inflate: an invalid block type");
        }
    }
    b.align();
    if (b.count >= 32) throw std::logic_error("inflate: more than a word buffered");
    // Bytes still buffered belong to the Adler-32; give them back.
    size_t at = b.at - size_t(b.count / 8);
    if (at + 4 > n - 2) throw std::runtime_error("inflate: the stream ends before its Adler-32");
    const uint8_t* q = p + 2 + at;
    const uint32_t want = uint32_t(q[0]) << 24 | uint32_t(q[1]) << 16 | uint32_t(q[2]) << 8 | q[3];
    uint32_t s1 = 1, s2 = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        s1 = (s1 + out[i]) % 65521u;
        s2 = (s2 + s1) % 65521u;
    }
    if ((s2 << 16 | s1) != want) throw std::runtime_error("inflate: the Adler-32 check fails");
    return out;
}

}  // namespace image
