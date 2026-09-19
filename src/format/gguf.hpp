#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "format/format.hpp"

// GGUF file format reader/writer, implemented from scratch.
// Implements GGUF v3 + GGML_TYPE_Q8_0 and GGML_TYPE_F32 tensors.
// Spec reference (llama.cpp gguf.h):
//   header: magic(u32) version(u32) tensor_count(u64) metadata_kv_count(u64)
//   metadata KVs: key(string) type(u32) value
//   padding to ALIGNMENT
//   tensor infos: name(string) n_dims(u32) dims(u64[n]) type(u32) offset(u64)
//                 + padding to ALIGNMENT after each
//   padding to ALIGNMENT
//   tensor data (each padded to ALIGNMENT), offset relative to data start
// string = u64 length + raw bytes (no terminator)

namespace gguf {

constexpr uint32_t MAGIC     = 0x46554747u; // 'GGUF'
constexpr uint32_t VERSION   = 3;
constexpr size_t   ALIGNMENT = 32;

constexpr uint32_t GGML_TYPE_F32  = 0;
constexpr uint32_t GGML_TYPE_Q4_0 = 2;
constexpr uint32_t GGML_TYPE_Q4_1 = 3;
constexpr uint32_t GGML_TYPE_Q8_0 = 8;
constexpr uint32_t GGML_TYPE_Q4_K = 12;
constexpr uint32_t GGML_TYPE_Q6_K = 14;
constexpr size_t   Q4_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q4_K_TYPESIZE = 144;  // 2 f16 + 12 packed 6-bit + 128 nibbles
constexpr size_t   Q4_1_BLOCK    = 32;   // values per block
constexpr size_t   Q4_1_TYPESIZE = 20;   // f16 scale + f16 min + 32 nibbles
constexpr size_t   Q6_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q6_K_TYPESIZE = 210;  // 128 low + 64 high + 16 scales + f16
constexpr size_t   Q4_0_BLOCK    = 32;   // values per block
constexpr size_t   Q4_0_TYPESIZE = 18;   // 2-byte f16 scale + 32 nibbles
constexpr size_t   Q8_0_BLOCK    = 32;   // values per block
constexpr size_t   Q8_0_TYPESIZE = 34;   // 2-byte f16 scale + 32 int8

// GGUF value types
enum : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12
};

struct MetaValue {
    uint32_t vtype = 0;
    uint64_t u = 0;      // unsigned / array element type
    int64_t  i = 0;
    uint32_t fb = 0;     // float32 bits
    double   f64 = 0;
    bool     b = false;
    std::string s;
    std::vector<MetaValue> arr;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne; // dims; ne[0] is fastest
    uint32_t type = 0;
    uint64_t offset = 0;

    uint64_t n_elements() const {
        uint64_t n = 1;
        for (auto d : ne) n *= d;
        return n;
    }
    uint64_t data_size() const {
        if (type == GGML_TYPE_Q8_0) return (n_elements() / Q8_0_BLOCK) * Q8_0_TYPESIZE;
        if (type == GGML_TYPE_Q4_0) return (n_elements() / Q4_0_BLOCK) * Q4_0_TYPESIZE;
        if (type == GGML_TYPE_Q4_1) return (n_elements() / Q4_1_BLOCK) * Q4_1_TYPESIZE;
        if (type == GGML_TYPE_Q4_K) return (n_elements() / Q4_K_BLOCK) * Q4_K_TYPESIZE;
        if (type == GGML_TYPE_Q6_K) return (n_elements() / Q6_K_BLOCK) * Q6_K_TYPESIZE;
        if (type == GGML_TYPE_F32)  return n_elements() * 4;
        throw std::runtime_error("unsupported tensor type in data_size");
    }
};

struct GGUFModel {
    std::vector<std::pair<std::string, MetaValue>> kv;
    std::vector<TensorInfo> tensors;
    // All tensor data in ONE contiguous allocation. This used to be a separate
    // heap block per tensor (399 of them on Qwen3-8B), which fragments the very
    // weight stream that decode is bandwidth bound on.
    std::vector<uint8_t> blob;
    std::vector<size_t> offsets;

    const uint8_t* tensor_data(size_t i) const { return blob.data() + offsets[i]; }
    uint8_t* tensor_data(size_t i) { return blob.data() + offsets[i]; }
    size_t tensor_bytes(size_t i) const { return (size_t)tensors[i].data_size(); }

    // Append one tensor's bytes. Callers that know the total should reserve
    // blob first; read_gguf sizes it exactly and reads in place instead.
    void add_tensor_data(const std::vector<uint8_t>& bytes) {
        offsets.push_back(blob.size());
        blob.insert(blob.end(), bytes.begin(), bytes.end());
    }
};

inline std::string read_string(std::istream& is) {
    uint64_t n;
    is.read((char*)&n, 8);
    std::string s;
    s.resize((size_t)n);
    if (n) is.read(&s[0], (std::streamsize)n);
    return s;
}

inline void write_string(std::ostream& os, const std::string& s) {
    uint64_t n = s.size();
    os.write((const char*)&n, 8);
    os.write(s.data(), (std::streamsize)s.size());
}

inline void pad_to(std::ostream& os, size_t align) {
    size_t pos = (size_t)os.tellp();
    size_t pad = (align - (pos % align)) % align;
    for (size_t k = 0; k < pad; k++) os.put(0);
}

// Reads a bare value (no type tag) given its GGUF value type.
// In GGUF, array elements are stored WITHOUT their own type tag: the array
// header carries a single element type, then each element is written as just
// its value. So we must read/write elements as bare typed values, not as full
// (tagged) metadata values.
inline MetaValue read_typed_value(std::istream& is, uint32_t t) {
    MetaValue v;
    v.vtype = t;
    switch (t) {
        case V_UINT8:  { uint8_t  x; is.read((char*)&x, 1); v.u = x; break; }
        case V_INT8:   { int8_t   x; is.read((char*)&x, 1); v.i = x; break; }
        case V_UINT16: { uint16_t x; is.read((char*)&x, 2); v.u = x; break; }
        case V_INT16:  { int16_t  x; is.read((char*)&x, 2); v.i = x; break; }
        case V_UINT32: { uint32_t x; is.read((char*)&x, 4); v.u = x; break; }
        case V_INT32:  { int32_t  x; is.read((char*)&x, 4); v.i = x; break; }
        case V_FLOAT32:{ float    x; is.read((char*)&x, 4); std::memcpy(&v.fb, &x, 4); break; }
        case V_BOOL:   { uint8_t  x; is.read((char*)&x, 1); v.b = x; break; }
        case V_STRING: v.s = read_string(is); break;
        case V_ARRAY: {
            uint32_t et; uint64_t cnt;
            is.read((char*)&et, 4);
            is.read((char*)&cnt, 8);
            v.u = et;
            v.arr.resize((size_t)cnt);
            for (auto& e : v.arr) e = read_typed_value(is, et);
            break;
        }
        case V_UINT64:  { uint64_t x; is.read((char*)&x, 8); v.u = x; break; }
        case V_INT64:   { int64_t  x; is.read((char*)&x, 8); v.i = x; break; }
        case V_FLOAT64: { double   x; is.read((char*)&x, 8); v.f64 = x; break; }
        default: throw std::runtime_error("unknown GGUF metadata value type");
    }
    return v;
}

inline MetaValue read_meta_value(std::istream& is) {
    uint32_t t;
    is.read((char*)&t, 4);
    return read_typed_value(is, t);
}

inline void write_typed_value(std::ostream& os, const MetaValue& v) {
    switch (v.vtype) {
        case V_UINT8:   { uint8_t  x = (uint8_t)v.u;  os.write((const char*)&x, 1); break; }
        case V_INT8:    { int8_t   x = (int8_t)v.i;   os.write((const char*)&x, 1); break; }
        case V_UINT16:  { uint16_t x = (uint16_t)v.u; os.write((const char*)&x, 2); break; }
        case V_INT16:   { int16_t  x = (int16_t)v.i;  os.write((const char*)&x, 2); break; }
        case V_UINT32:  { uint32_t x = (uint32_t)v.u; os.write((const char*)&x, 4); break; }
        case V_INT32:   { int32_t  x = (int32_t)v.i;  os.write((const char*)&x, 4); break; }
        case V_FLOAT32: { float x; std::memcpy(&x, &v.fb, 4); os.write((const char*)&x, 4); break; }
        case V_BOOL:    { uint8_t  x = v.b ? 1 : 0;   os.write((const char*)&x, 1); break; }
        case V_STRING:  write_string(os, v.s); break;
        case V_ARRAY: {
            uint32_t et = (uint32_t)v.u;
            uint64_t cnt = v.arr.size();
            os.write((const char*)&et, 4);
            os.write((const char*)&cnt, 8);
            for (const auto& e : v.arr) write_typed_value(os, e);
            break;
        }
        case V_UINT64:  { uint64_t x = v.u; os.write((const char*)&x, 8); break; }
        case V_INT64:   { int64_t  x = v.i; os.write((const char*)&x, 8); break; }
        case V_FLOAT64: { double   x = v.f64; os.write((const char*)&x, 8); break; }
        default: throw std::runtime_error("unknown GGUF metadata value type");
    }
}

inline void write_meta_value(std::ostream& os, const MetaValue& v) {
    os.write((const char*)&v.vtype, 4);
    write_typed_value(os, v);
}

inline void write_gguf(const GGUFModel& m, const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("cannot open file for writing: " + path);

    uint32_t magic = MAGIC;
    uint32_t ver   = VERSION;
    uint64_t ntc   = m.tensors.size();
    uint64_t nkv   = m.kv.size();
    os.write((const char*)&magic, 4);
    os.write((const char*)&ver,   4);
    os.write((const char*)&ntc,   8);
    os.write((const char*)&nkv,   8);

    for (const auto& kv : m.kv) {
        write_string(os, kv.first);
        write_meta_value(os, kv.second);
    }

    // Tensor infos are packed contiguously (no padding between them).
    uint64_t data_offset = 0; // aligned start of each tensor's data
    for (const auto& t : m.tensors) {
        write_string(os, t.name);
        uint32_t nd = (uint32_t)t.ne.size();
        os.write((const char*)&nd, 4);
        for (uint64_t d : t.ne) os.write((const char*)&d, 8);
        os.write((const char*)&t.type, 4);
        os.write((const char*)&data_offset, 8);
        uint64_t end = data_offset + t.data_size();
        data_offset = (end + (ALIGNMENT - 1)) & ~(uint64_t)(ALIGNMENT - 1);
    }
    pad_to(os, ALIGNMENT);

    for (size_t i = 0; i < m.tensors.size(); i++) {
        os.write((const char*)m.tensor_data(i), (std::streamsize)m.tensor_bytes(i));
        pad_to(os, ALIGNMENT);
    }
}

inline GGUFModel read_gguf(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("cannot open file: " + path);

    GGUFModel m;
    uint32_t magic;
    uint32_t ver;
    uint64_t ntc, nkv;
    is.read((char*)&magic, 4);
    if (magic != MAGIC) throw std::runtime_error("not a GGUF file (bad magic)");
    is.read((char*)&ver, 4);
    is.read((char*)&ntc, 8);
    is.read((char*)&nkv, 8);

    for (uint64_t k = 0; k < nkv; k++) {
        std::string key = read_string(is);
        m.kv.emplace_back(std::move(key), read_meta_value(is));
    }

    // Tensor infos are packed contiguously (no padding between them).
    for (uint64_t i = 0; i < ntc; i++) {
        TensorInfo t;
        t.name = read_string(is);
        uint32_t nd;
        is.read((char*)&nd, 4);
        t.ne.resize(nd);
        for (uint32_t d = 0; d < nd; d++) is.read((char*)&t.ne[d], 8);
        is.read((char*)&t.type, 4);
        is.read((char*)&t.offset, 8);
        m.tensors.push_back(std::move(t));
    }
    // Only the tensor-data section start is aligned to ALIGNMENT.
    { size_t pos = (size_t)is.tellg(); size_t pad = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT; is.seekg(pos + pad); }
    size_t data_start = (size_t)is.tellg();

    // Size the blob exactly, then read each tensor straight into place: no
    // per-tensor temporary and no reallocation of an 8 GB buffer.
    size_t total = 0;
    m.offsets.reserve(m.tensors.size());
    for (const auto& t : m.tensors) { m.offsets.push_back(total); total += (size_t)t.data_size(); }
    m.blob.resize(total);
    for (size_t i = 0; i < m.tensors.size(); i++) {
        is.seekg(data_start + m.tensors[i].offset);
        is.read((char*)m.tensor_data(i), (std::streamsize)m.tensor_bytes(i));
    }
    return m;
}

// GGUF is the reference implementation of the format::ModelFormat interface.
// It wraps a GGUFModel (already read into memory) and exposes its tensors and
// metadata through the format-agnostic view.
class GGUFFormat final : public format::ModelFormat {
public:
    explicit GGUFFormat(gguf::GGUFModel m) : m_(std::move(m)) {}

    std::vector<format::Tensor> tensors() const override {
        std::vector<format::Tensor> out;
        out.reserve(m_.tensors.size());
        for (const auto& t : m_.tensors) out.push_back({ t.name, t.ne, t.type });
        return out;
    }

    std::string metadata_string(const std::string& key) const override {
        for (const auto& kv : m_.kv)
            if (kv.first == key && kv.second.vtype == gguf::V_STRING) return kv.second.s;
        return "";
    }

    uint64_t metadata_u64(const std::string& key) const override {
        for (const auto& kv : m_.kv) {
            if (kv.first != key) continue;
            switch (kv.second.vtype) {
                case gguf::V_UINT8:  return kv.second.u;
                case gguf::V_UINT16: return kv.second.u;
                case gguf::V_UINT32: return kv.second.u;
                case gguf::V_UINT64: return kv.second.u;
                case gguf::V_INT8:   return (uint64_t)kv.second.i;
                case gguf::V_INT16:  return (uint64_t)kv.second.i;
                case gguf::V_INT32:  return (uint64_t)kv.second.i;
                case gguf::V_INT64:  return (uint64_t)kv.second.i;
                default: return 0;
            }
        }
        return 0;
    }

private:
    gguf::GGUFModel m_;
};

} // namespace gguf

// Auto-detect the format from the file header and open it. Currently only GGUF
// is implemented; the magic check is the extension point for future formats.
inline format::ModelFormatPtr format::open(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("cannot open file: " + path);
    uint32_t magic;
    is.read((char*)&magic, 4);
    if (magic == gguf::MAGIC) return std::make_shared<gguf::GGUFFormat>(gguf::read_gguf(path));
    return nullptr;
}
