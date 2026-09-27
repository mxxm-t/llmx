#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"

// Typed reads of a GGUF file's metadata with the range rules and refusal texts every architecture's reader shares.

namespace infer::metadata {

// A positive integer up to INT_MAX from any of the four integer types; an absent key takes `fallback`, and is refused when that is zero.
inline int integer(const gguf::GGUFModel& m, const std::string& k, int fallback = 0) {
    const auto* v = m.find(k);
    if (!v) {
        if (fallback) return fallback;
        throw std::runtime_error("inference: missing metadata " + k);
    }
    uint64_t n;
    if (v->vtype == gguf::V_UINT32 || v->vtype == gguf::V_UINT64) {
        n = v->u;
    } else if (v->vtype == gguf::V_INT32 || v->vtype == gguf::V_INT64) {
        if (v->i <= 0) throw std::runtime_error("inference: invalid positive integer " + k);
        n = uint64_t(v->i);
    } else {
        throw std::runtime_error("inference: invalid integer type " + k);
    }
    if (!n || n > uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("inference: integer outside supported range " + k);
    return int(n);
}

// A finite positive F32 or F64 that is a nonzero F32; an absent key takes `fallback`.
inline double real(const gguf::GGUFModel& m, const std::string& k, float fallback) {
    const auto* v = m.find(k);
    if (!v) return fallback;
    double n;
    if (v->vtype == gguf::V_FLOAT32) {
        float value;
        std::memcpy(&value, &v->fb, sizeof(value));
        n = value;
    } else if (v->vtype == gguf::V_FLOAT64) {
        n = v->f64;
    } else {
        throw std::runtime_error("inference: invalid floating-point type " + k);
    }
    if (!std::isfinite(n) || n <= 0 || n > std::numeric_limits<float>::max())
        throw std::runtime_error("inference: invalid positive float " + k);
    const float value = float(n);
    if (value == 0) throw std::runtime_error("inference: float underflow " + k);
    return n;
}

// A string key that may be absent or hold only the one value the reader supports.
inline void option(const gguf::GGUFModel& m, const std::string& k, const std::string& supported) {
    const auto* v = m.find(k);
    if (v && (v->vtype != gguf::V_STRING || v->s != supported))
        throw std::runtime_error("inference: unsupported metadata " + k);
}

// An enumerated UINT32 key that may be absent or hold only the one value the reader supports, refused otherwise with the reader's text.
inline void choice(const gguf::GGUFModel& m, const std::string& k, uint64_t supported, const std::string& refusal) {
    const auto* v = m.find(k);
    if (v && (v->vtype != gguf::V_UINT32 || v->u != supported)) throw std::runtime_error(refusal);
}

// A boolean; an absent key takes `fallback`.
inline bool boolean(const gguf::GGUFModel& m, const std::string& k, bool fallback) {
    const auto* v = m.find(k);
    if (!v) return fallback;
    if (v->vtype != gguf::V_BOOL) throw std::runtime_error("inference: invalid boolean type " + k);
    return v->b;
}

// A value of any of the four integer types from 0 up to INT_MAX, which `count` and `counts` read.
inline int count_value(const gguf::MetaValue& v, const std::string& k) {
    int64_t n;
    if (v.vtype == gguf::V_UINT32 || v.vtype == gguf::V_UINT64) {
        if (v.u > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error("inference: integer outside supported range " + k);
        n = int64_t(v.u);
    } else if (v.vtype == gguf::V_INT32 || v.vtype == gguf::V_INT64) {
        if (v.i < 0 || v.i > int64_t(std::numeric_limits<int>::max())) throw std::runtime_error("inference: integer outside supported range " + k);
        n = v.i;
    } else {
        throw std::runtime_error("inference: invalid integer type " + k);
    }
    return int(n);
}

// A count, which may be zero, from any of the four integer types up to INT_MAX; an absent key takes `fallback`.
inline int count(const gguf::GGUFModel& m, const std::string& k, int fallback) {
    const auto* v = m.find(k);
    return v ? count_value(*v, k) : fallback;
}

// An array of counts, each as `count` reads one; an absent key gives none.
inline std::vector<int> counts(const gguf::GGUFModel& m, const std::string& k) {
    const auto* v = m.find(k);
    if (!v) return {};
    if (v->vtype != gguf::V_ARRAY) throw std::runtime_error("inference: invalid array type " + k);
    std::vector<int> out;
    for (const gguf::MetaValue& e : v->arr) out.push_back(count_value(e, k));
    return out;
}

// An array of booleans; an absent key gives none.
inline std::vector<bool> booleans(const gguf::GGUFModel& m, const std::string& k) {
    const auto* v = m.find(k);
    if (!v) return {};
    if (v->vtype != gguf::V_ARRAY) throw std::runtime_error("inference: invalid array type " + k);
    std::vector<bool> out;
    for (const gguf::MetaValue& e : v->arr) {
        if (e.vtype != gguf::V_BOOL) throw std::runtime_error("inference: invalid boolean type " + k);
        out.push_back(e.b);
    }
    return out;
}

} // namespace infer::metadata
