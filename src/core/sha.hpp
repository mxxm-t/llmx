#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace core {

// SHA-1 is used only for the Hub's Git blob identity; LFS payloads use SHA-256.
class Sha {
    bool sha256_;
    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> pending_{};
    uint64_t bytes_ = 0;
    size_t used_ = 0;

    static uint32_t rotate(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* data) {
        uint32_t w[80];
        for (size_t i = 0; i < 16; ++i)
            w[i] = (uint32_t(data[4*i]) << 24) | (uint32_t(data[4*i+1]) << 16) |
                   (uint32_t(data[4*i+2]) << 8) | data[4*i+3];
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4];
        if (!sha256_) {
            for (size_t i = 16; i < 80; ++i) w[i] = rotate(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 31);
            for (size_t i = 0; i < 80; ++i) {
                const uint32_t f = i < 20 ? ((b & c) | (~b & d)) :
                    i < 40 || i >= 60 ? (b ^ c ^ d) : ((b & c) | (b & d) | (c & d));
                const uint32_t k = i < 20 ? 0x5a827999u : i < 40 ? 0x6ed9eba1u :
                                   i < 60 ? 0x8f1bbcdcu : 0xca62c1d6u;
                const uint32_t next = rotate(a, 27) + f + e + k + w[i];
                e = d; d = c; c = rotate(b, 2); b = a; a = next;
            }
        } else {
            static constexpr uint32_t k[64] = {
                0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
                0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
                0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
                0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
                0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
                0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
                0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
                0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
            };
            for (size_t i = 16; i < 64; ++i) {
                const uint32_t s0 = rotate(w[i-15], 7) ^ rotate(w[i-15], 18) ^ (w[i-15] >> 3);
                const uint32_t s1 = rotate(w[i-2], 17) ^ rotate(w[i-2], 19) ^ (w[i-2] >> 10);
                w[i] = w[i-16] + s0 + w[i-7] + s1;
            }
            uint32_t f = state_[5], g = state_[6], h = state_[7];
            for (size_t i = 0; i < 64; ++i) {
                const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
                const uint32_t t1 = h + s1 + ((e & f) ^ (~e & g)) + k[i] + w[i];
                const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
                const uint32_t t2 = s0 + ((a & b) ^ (a & c) ^ (b & c));
                h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            }
            state_[5] += f; state_[6] += g; state_[7] += h;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d; state_[4] += e;
    }

public:
    explicit Sha(bool sha256) : sha256_(sha256), state_(sha256 ?
        std::array<uint32_t, 8>{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19} :
        std::array<uint32_t, 8>{0x67452301,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0,0,0,0}) {}

    void update(const void* data, size_t length) {
        if (!length) return;
        if (length > std::numeric_limits<uint64_t>::max() / 8 - bytes_)
            throw std::runtime_error("SHA input too large");
        bytes_ += length;
        const auto* in = static_cast<const uint8_t*>(data);
        if (used_) {
            const size_t count = std::min(length, 64 - used_);
            if (count) std::memcpy(pending_.data() + used_, in, count);
            used_ += count; in += count; length -= count;
            if (used_ == 64) { block(pending_.data()); used_ = 0; }
        }
        while (length >= 64) { block(in); in += 64; length -= 64; }
        if (length) { std::memcpy(pending_.data(), in, length); used_ = length; }
    }

    std::string hex() const {
        Sha copy = *this;
        const uint64_t bits = bytes_ * 8;
        copy.pending_[copy.used_++] = 0x80;
        if (copy.used_ > 56) {
            std::fill(copy.pending_.begin() + copy.used_, copy.pending_.end(), uint8_t(0));
            copy.block(copy.pending_.data()); copy.used_ = 0;
        }
        std::fill(copy.pending_.begin() + copy.used_, copy.pending_.begin() + 56, uint8_t(0));
        for (unsigned i = 0; i < 8; ++i) copy.pending_[63-i] = uint8_t(bits >> (8*i));
        copy.block(copy.pending_.data());
        static constexpr char digits[] = "0123456789abcdef";
        std::string out;
        for (size_t i = 0; i < (sha256_ ? 8u : 5u); ++i)
            for (int shift = 28; shift >= 0; shift -= 4) out += digits[(copy.state_[i] >> shift) & 15];
        return out;
    }
};

} // namespace core
