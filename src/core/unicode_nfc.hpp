#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/utf8.hpp"

// Unicode 16.0.0 Normalization Form C (UAX #15) over UTF-8, as a tokenizer's NFC normalizer applies it (docs/src/core-unicode_nfc.md).

namespace unicode_nfc {

struct CombiningRun { uint32_t first, last; uint8_t ccc; };
struct Decomposition { uint32_t cp, first, second; };
struct Composition { uint32_t first, second, composite; };
struct Run { uint32_t first, last; };

#include "core/unicode_nfc_table.inc"

constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kSCount = kLCount * kVCount * kTCount;

template <class T, size_t N>
inline const T* in_runs(const T (&runs)[N], uint32_t cp) {
    const T* it = std::lower_bound(runs, runs + N, cp, [](const T& r, uint32_t c) { return r.last < c; });
    return it != runs + N && it->first <= cp ? it : nullptr;
}

inline uint8_t ccc(uint32_t cp) {
    if (cp < 0x300) return 0;
    const CombiningRun* r = in_runs(kCombining, cp);
    return r ? r->ccc : 0;
}

inline void decompose(uint32_t cp, std::vector<uint32_t>& out) {
    if (cp - kSBase < kSCount) {
        const uint32_t s = cp - kSBase;
        out.push_back(kLBase + s / (kVCount * kTCount));
        out.push_back(kVBase + s % (kVCount * kTCount) / kTCount);
        if (s % kTCount) out.push_back(kTBase + s % kTCount);
        return;
    }
    const auto* end = kDecompositions + sizeof(kDecompositions) / sizeof(kDecompositions[0]);
    const auto* it = std::lower_bound(kDecompositions, end, cp, [](const Decomposition& d, uint32_t c) { return d.cp < c; });
    if (it == end || it->cp != cp) {
        out.push_back(cp);
        return;
    }
    decompose(it->first, out);
    if (it->second) decompose(it->second, out);
}

// The primary composite of `a` then `b`, or 0.
inline uint32_t compose(uint32_t a, uint32_t b) {
    if (a - kLBase < kLCount && b - kVBase < kVCount) return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a - kSBase < kSCount && (a - kSBase) % kTCount == 0 && b - kTBase - 1 < kTCount - 1) return a + (b - kTBase);
    const auto* end = kCompositions + sizeof(kCompositions) / sizeof(kCompositions[0]);
    const auto* it = std::lower_bound(kCompositions, end, std::make_pair(a, b), [](const Composition& c, const std::pair<uint32_t, uint32_t>& k) {
        return c.first < k.first || (c.first == k.first && c.second < k.second);
    });
    return it != end && it->first == a && it->second == b ? it->composite : 0;
}

// NFC of the code points of one valid run: full canonical decomposition, canonical ordering, then canonical composition.
inline void normalize_run(const std::vector<uint32_t>& in, std::string& out) {
    std::vector<uint32_t> d;
    for (uint32_t cp : in) decompose(cp, d);
    for (size_t i = 1; i < d.size(); ++i) {
        const uint8_t c = ccc(d[i]);
        if (!c) continue;
        for (size_t j = i; j > 0 && ccc(d[j - 1]) > c; --j) std::swap(d[j - 1], d[j]);
    }
    std::vector<uint32_t> o;
    size_t starter = SIZE_MAX;
    for (uint32_t cp : d) {
        const uint8_t c = ccc(cp);
        if (starter != SIZE_MAX && (o.size() - 1 == starter || ccc(o.back()) < c))
            if (const uint32_t composite = compose(o[starter], cp)) {
                o[starter] = composite;
                continue;
            }
        if (!c) starter = o.size();
        o.push_back(cp);
    }
    for (uint32_t cp : o) out += utf8::encode(cp);
}

// The code point of the valid sequence of `n` bytes at s[i].
inline uint32_t decode(const std::string& s, size_t i, size_t n) {
    static constexpr unsigned char kLeadMask[] = {0, 0x7F, 0x1F, 0x0F, 0x07};
    uint32_t cp = static_cast<unsigned char>(s[i]) & kLeadMask[n];
    for (size_t k = 1; k < n; ++k) cp = cp << 6 | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    return cp;
}

// `text` in NFC: each run of valid UTF-8 is normalized and an invalid byte is kept as it is, ending the run before it.
// Text the quick check passes (every code point quick-check Yes, combining classes in order) is returned unchanged after one scan.
inline std::string normalize(std::string text) {
    size_t i = 0;
    while (i < text.size() && static_cast<unsigned char>(text[i]) < 0x80) ++i;
    if (i == text.size()) return text;
    bool clean = true;
    uint8_t last = 0;
    for (size_t k = i; k < text.size() && clean;) {
        const size_t n = utf8::valid_length(text, k);
        if (!n) { last = 0; ++k; continue; }
        const uint32_t cp = decode(text, k, n);
        const uint8_t c = ccc(cp);
        if ((c && last > c) || (cp >= 0x300 && in_runs(kNotQuick, cp))) clean = false;
        last = c;
        k += n;
    }
    if (clean) return text;
    std::string out;
    out.reserve(text.size());
    std::vector<uint32_t> run;
    for (size_t k = 0; k < text.size();) {
        const size_t n = utf8::valid_length(text, k);
        if (n) {
            run.push_back(decode(text, k, n));
            k += n;
            continue;
        }
        normalize_run(run, out);
        run.clear();
        out += text[k++];
    }
    normalize_run(run, out);
    return out;
}

} // namespace unicode_nfc
