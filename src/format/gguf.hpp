#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <unordered_set>

#include "format/format.hpp"
#include "format/mapped_file.hpp"
#include "core/host_memory.hpp"

// GGUF file format reader/writer, implemented from scratch.
// Implements GGUF v3 and the tensor types listed below.
// File layout:
//   header: magic(u32) version(u32) tensor_count(u64) metadata_kv_count(u64)
//   metadata KVs: key(string) type(u32) value
//   tensor infos: name(string) n_dims(u32) dims(u64[n]) type(u32) offset(u64)
//   padding to ALIGNMENT
//   tensor data (each padded to ALIGNMENT), offset relative to data start
// string = u64 length + raw bytes (no terminator)

namespace gguf {

constexpr uint32_t MAGIC     = 0x46554747u; // 'GGUF'
constexpr uint32_t VERSION   = 3;
constexpr size_t   ALIGNMENT = 32;
constexpr unsigned MAX_ARRAY_DEPTH = 256;

inline uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw std::runtime_error("GGUF size addition overflow");
    return a + b;
}

inline uint64_t checked_multiply(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a)
        throw std::runtime_error("GGUF size multiplication overflow");
    return a * b;
}

inline uint64_t aligned_size(uint64_t value, uint64_t alignment) {
    return checked_add(value, (alignment - value % alignment) % alignment);
}

constexpr uint32_t GGML_TYPE_F32  = 0;
constexpr uint32_t GGML_TYPE_Q4_0 = 2;
constexpr uint32_t GGML_TYPE_Q4_1 = 3;
constexpr uint32_t GGML_TYPE_Q8_0 = 8;
constexpr uint32_t GGML_TYPE_Q4_K = 12;
constexpr uint32_t GGML_TYPE_Q5_K = 13;
constexpr uint32_t GGML_TYPE_Q6_K = 14;
constexpr size_t   Q5_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q5_K_TYPESIZE = 176;  // Q4_K plus 32 bytes of fifth bits
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
        if (std::find(ne.begin(), ne.end(), uint64_t(0)) != ne.end()) return 0;
        uint64_t n = 1;
        for (auto d : ne) n = checked_multiply(n, d);
        return n;
    }
    uint64_t data_size() const {
        uint64_t block, bytes;
        switch (type) {
            case GGML_TYPE_F32:  block = 1;          bytes = 4;               break;
            case GGML_TYPE_Q8_0: block = Q8_0_BLOCK; bytes = Q8_0_TYPESIZE;     break;
            case GGML_TYPE_Q4_0: block = Q4_0_BLOCK; bytes = Q4_0_TYPESIZE;     break;
            case GGML_TYPE_Q4_1: block = Q4_1_BLOCK; bytes = Q4_1_TYPESIZE;     break;
            case GGML_TYPE_Q4_K: block = Q4_K_BLOCK; bytes = Q4_K_TYPESIZE;     break;
            case GGML_TYPE_Q5_K: block = Q5_K_BLOCK; bytes = Q5_K_TYPESIZE;     break;
            case GGML_TYPE_Q6_K: block = Q6_K_BLOCK; bytes = Q6_K_TYPESIZE;     break;
            default: throw std::runtime_error("unsupported tensor type in data_size");
        }
        if (block != 1 && (ne.empty() || ne[0] % block))
            throw std::runtime_error("GGUF quantized row is not a whole number of blocks");
        return checked_multiply(n_elements() / block, bytes);
    }
};

struct GGUFModel {
    std::vector<std::pair<std::string, MetaValue>> kv;
    std::vector<TensorInfo> tensors;
    // All tensor data, addressed by `offsets`: an in-memory model's in one allocation, a file's data section mapped read-only, one segment per shard.
    // One heap block per tensor would fragment the weight stream that decode is bandwidth bound on, and mapping shards rather than copying them loads a model larger than host memory.
    std::vector<uint8_t> blob;
    struct Segment {
        std::shared_ptr<const format::MappedFile> file;
        size_t start = 0;   // where the data section starts in the file
        size_t base = 0;    // the offset its first byte has among `offsets`; segments follow one another in offset order
        size_t size = 0;
    };
    std::vector<Segment> segments;
    std::vector<size_t> offsets;

    // Drop the tensor bytes, keeping the metadata and the tensor table. For a caller whose model no longer reads them in place, which is any model whose every weight a copying backend took. tensor_data is invalid afterwards.
    void release_payload() {
        std::vector<uint8_t>().swap(blob);
        segments.clear();
    }

    // The extent `offsets` address, whichever holds the tensor bytes.
    size_t payload_size() const { return segments.empty() ? blob.size() : segments.back().base + segments.back().size; }

    const uint8_t* tensor_data(size_t i) const {
        if (segments.empty()) return blob.data() + offsets[i];
        const Segment& s = segment_of(offsets[i]);
        return s.file->data() + s.start + (offsets[i] - s.base);
    }
    // A mapped model's tensor no host reads in place: its pages leave the host's working set first (MappedFile::drop). Nothing for an in-memory model.
    void drop_pages(size_t i) const {
        if (!segments.empty()) segment_of(offsets[i]).file->drop(tensor_data(i), tensor_bytes(i));
    }
    // The same bytes; a mapped model's are read-only memory, so only an in-memory model may be written through this.
    size_t tensor_bytes(size_t i) const { return (size_t)tensors[i].data_size(); }

    // Append one tensor's bytes.
    // Callers that know the total should reserve blob first; read_gguf sizes it exactly and reads in place instead.
    void add_tensor_data(const std::vector<uint8_t>& bytes) {
        blob.resize((blob.size() + alignof(float) - 1) / alignof(float) * alignof(float));
        offsets.push_back(blob.size());
        blob.insert(blob.end(), bytes.begin(), bytes.end());
    }

    // The value stored under `key`, or null. A file repeats no key; the reader refuses one that does.
    const MetaValue* find(const std::string& key) const {
        for (const auto& item : kv)
            if (item.first == key) return &item.second;
        return nullptr;
    }

private:
    // The segment holding `offset`: the last whose base is not past it, so a zero-sized tensor at a boundary takes the next.
    const Segment& segment_of(size_t offset) const {
        auto it = std::upper_bound(segments.begin(), segments.end(), offset, [](size_t o, const Segment& s) { return o < s.base; });
        return *(it == segments.begin() ? it : it - 1);
    }
};

inline uint32_t file_alignment(const GGUFModel& m) {
    const MetaValue* value = m.find("general.alignment");
    if (!value) return uint32_t(ALIGNMENT);
    if (value->vtype != V_UINT32 || !value->u || value->u % 8 || value->u > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("invalid GGUF general.alignment");
    return uint32_t(value->u);
}

class Reader {
    std::istream& stream_;
    uint64_t size_, position_ = 0;
public:
    explicit Reader(std::istream& stream) : stream_(stream) {
        stream_.seekg(0, std::ios::end);
        const std::streamoff size = stream_.tellg();
        if (size < 0) throw std::ios_base::failure("cannot determine GGUF file size");
        size_ = uint64_t(size);
        stream_.seekg(0);
    }

    uint64_t size() const { return size_; }
    uint64_t position() const { return position_; }
    uint64_t remaining() const { return size_ - position_; }

    void require(uint64_t bytes) const {
        if (bytes > remaining()) throw std::ios_base::failure("GGUF field exceeds file extent");
    }

    void read(char* data, uint64_t bytes) {
        require(bytes);
        if (bytes > uint64_t(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("GGUF field exceeds stream size limit");
        stream_.read(data, std::streamsize(bytes));
        position_ += bytes;
    }
};

inline std::string read_string(Reader& is) {
    uint64_t n;
    is.read((char*)&n, 8);
    is.require(n);
    std::string s;
    if (n > s.max_size()) throw std::runtime_error("GGUF string exceeds allocation limit");
    s.resize((size_t)n);
    if (n) is.read(&s[0], n);
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
// In GGUF, array elements are stored WITHOUT their own type tag: the array header carries a single element type, then each element is written as just its value.
// So we must read/write elements as bare typed values, not as full (tagged) metadata values.
inline uint64_t minimum_value_size(uint32_t type) {
    switch (type) {
        case V_UINT8: case V_INT8: case V_BOOL: return 1;
        case V_UINT16: case V_INT16: return 2;
        case V_UINT32: case V_INT32: case V_FLOAT32: return 4;
        case V_UINT64: case V_INT64: case V_FLOAT64: case V_STRING: return 8;
        case V_ARRAY: return 12;
        default: throw std::runtime_error("unknown GGUF metadata value type");
    }
}

inline MetaValue read_typed_value(Reader& is, uint32_t t, unsigned depth = 0) {
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
            if (depth >= MAX_ARRAY_DEPTH) throw std::runtime_error("GGUF array nesting limit exceeded");
            uint32_t et; uint64_t cnt;
            is.read((char*)&et, 4);
            is.read((char*)&cnt, 8);
            const uint64_t width = minimum_value_size(et);
            if (cnt > is.remaining() / width)
                throw std::ios_base::failure("GGUF array exceeds file extent");
            if (cnt > v.arr.max_size()) throw std::runtime_error("GGUF array exceeds allocation limit");
            v.u = et;
            v.arr.resize((size_t)cnt);
            for (auto& e : v.arr) e = read_typed_value(is, et, depth + 1);
            break;
        }
        case V_UINT64:  { uint64_t x; is.read((char*)&x, 8); v.u = x; break; }
        case V_INT64:   { int64_t  x; is.read((char*)&x, 8); v.i = x; break; }
        case V_FLOAT64: { double   x; is.read((char*)&x, 8); v.f64 = x; break; }
        default: throw std::runtime_error("unknown GGUF metadata value type");
    }
    return v;
}

inline MetaValue read_meta_value(Reader& is) {
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
    const uint32_t alignment = file_alignment(m);
    std::ofstream os(std::filesystem::u8path(path), std::ios::binary);
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
        data_offset = aligned_size(checked_add(data_offset, t.data_size()), alignment);
    }
    pad_to(os, alignment);

    for (size_t i = 0; i < m.tensors.size(); i++) {
        os.write((const char*)m.tensor_data(i), (std::streamsize)m.tensor_bytes(i));
        pad_to(os, alignment);
    }
}

namespace detail {

inline uint64_t read_header(std::ifstream& is, GGUFModel& m) {
    Reader reader(is);
    uint32_t magic;
    uint32_t ver;
    uint64_t ntc, nkv;
    reader.read((char*)&magic, 4);
    if (magic != MAGIC) throw std::runtime_error("not a GGUF file (bad magic)");
    reader.read((char*)&ver, 4);
    if (ver != VERSION) throw std::runtime_error("unsupported GGUF version");
    reader.read((char*)&ntc, 8);
    reader.read((char*)&nkv, 8);
    if (ntc > reader.remaining() / 24 || nkv > reader.remaining() / 13 ||
        ntc > m.tensors.max_size() || nkv > m.kv.max_size())
        throw std::runtime_error("GGUF entry count exceeds file or allocation limit");

    std::unordered_set<std::string> keys;
    for (uint64_t k = 0; k < nkv; k++) {
        std::string key = read_string(reader);
        if (!keys.insert(key).second) throw std::runtime_error("duplicate GGUF metadata: " + key);
        m.kv.emplace_back(std::move(key), read_meta_value(reader));
    }
    const uint32_t alignment = file_alignment(m);

    // Tensor infos are packed contiguously (no padding between them).
    for (uint64_t i = 0; i < ntc; i++) {
        TensorInfo t;
        t.name = read_string(reader);
        uint32_t nd;
        reader.read((char*)&nd, 4);
        if (nd > 4) throw std::runtime_error("unsupported GGUF tensor rank");
        reader.require(uint64_t(nd) * 8 + 12);
        t.ne.resize(nd);
        for (uint32_t d = 0; d < nd; d++) reader.read((char*)&t.ne[d], 8);
        reader.read((char*)&t.type, 4);
        reader.read((char*)&t.offset, 8);
        m.tensors.push_back(std::move(t));
    }
    const uint64_t data_start = aligned_size(reader.position(), alignment);
    if (data_start > reader.size()) throw std::ios_base::failure("GGUF data section exceeds file extent");
    const uint64_t available = reader.size() - data_start;

    for (const auto& t : m.tensors) {
        const uint64_t bytes = t.data_size();
        if (t.offset % alignment) throw std::runtime_error("unaligned GGUF tensor offset");
        if (t.offset > available || bytes > available - t.offset)
            throw std::ios_base::failure("GGUF tensor exceeds file extent");
    }
    return data_start;
}

struct Input {
    std::string path;
    std::ifstream stream;
    uint64_t data_start;
    size_t begin, end;

    Input(const std::string& file, GGUFModel& header, size_t first)
        : path(file), stream(std::filesystem::u8path(file), std::ios::binary), begin(first) {
        if (!stream) throw std::runtime_error("cannot open file: " + path);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        data_start = read_header(stream, header);
        const uint64_t total = checked_add(first, header.tensors.size());
        if (total > header.tensors.max_size())
            throw std::runtime_error("GGUF tensor count exceeds allocation limit");
        end = size_t(total);
    }
};

struct Split {
    uint16_t no = 0, count = 1;
    int32_t tensors = 0;
    bool present = false;
};

inline Split split_info(const GGUFModel& m) {
    const MetaValue* values[3] = {m.find("split.no"), m.find("split.count"), m.find("split.tensors.count")};
    if (!values[0] && !values[1] && !values[2]) return {};
    if (!values[0] || !values[1] || !values[2] ||
        values[0]->vtype != V_UINT16 || values[1]->vtype != V_UINT16 ||
        values[2]->vtype != V_INT32 || !values[1]->u ||
        values[0]->u >= values[1]->u || values[2]->i < 0)
        throw std::runtime_error("invalid GGUF split metadata");
    return {uint16_t(values[0]->u), uint16_t(values[1]->u), int32_t(values[2]->i), true};
}

inline bool equal_value(const MetaValue& a, const MetaValue& b) {
    if (a.vtype != b.vtype) return false;
    switch (a.vtype) {
        case V_UINT8: case V_UINT16: case V_UINT32: case V_UINT64: return a.u == b.u;
        case V_INT8: case V_INT16: case V_INT32: case V_INT64: return a.i == b.i;
        case V_FLOAT32: return a.fb == b.fb;
        case V_FLOAT64: return std::memcmp(&a.f64, &b.f64, sizeof(double)) == 0;
        case V_BOOL: return a.b == b.b;
        case V_STRING: return a.s == b.s;
        case V_ARRAY:
            if (a.u != b.u || a.arr.size() != b.arr.size()) return false;
            for (size_t i = 0; i < a.arr.size(); ++i)
                if (!equal_value(a.arr[i], b.arr[i])) return false;
            return true;
        default: return false;
    }
}

inline std::string split_suffix(uint16_t no, uint16_t count) {
    const std::string index = std::to_string(unsigned(no) + 1);
    const std::string total = std::to_string(count);
    return "-" + std::string(5 - index.size(), '0') + index + "-of-" +
           std::string(5 - total.size(), '0') + total + ".gguf";
}

} // namespace detail

inline GGUFModel read_gguf(const std::string& path, const format::LoadProgress& progress = {}) {
    GGUFModel m;
    std::vector<detail::Input> files;
    files.emplace_back(path, m, 0);
    const auto split = detail::split_info(m);
    if (split.present) {
        if (split.no) throw std::runtime_error("open the first GGUF shard (split.no must be zero)");
        std::string prefix;
        if (split.count > 1) {
            const std::string suffix = detail::split_suffix(0, split.count);
            if (path.size() < suffix.size() || path.compare(path.size() - suffix.size(), suffix.size(), suffix))
                throw std::runtime_error("GGUF shard filename must end in " + suffix);
            prefix = path.substr(0, path.size() - suffix.size());
        }
        std::unordered_set<std::string> names;
        auto check_tensors = [&](const GGUFModel& header, size_t total) {
            if (total > uint64_t(split.tensors))
                throw std::runtime_error("GGUF split tensor count mismatch");
            for (const auto& t : header.tensors)
                if (!names.insert(t.name).second)
                    throw std::runtime_error("duplicate GGUF shard tensor: " + t.name);
        };
        check_tensors(m, m.tensors.size());
        for (uint16_t index = 1; index < split.count; ++index) {
            GGUFModel header;
            files.emplace_back(prefix + detail::split_suffix(index, split.count), header, m.tensors.size());
            const auto next = detail::split_info(header);
            if (!next.present || next.no != index || next.count != split.count || next.tensors != split.tensors)
                throw std::runtime_error("inconsistent GGUF split metadata");
            for (const auto& kv : header.kv) {
                if (kv.first == "split.no" || kv.first == "general.alignment") continue;
                const MetaValue* first_value = m.find(kv.first);
                if (!first_value || !detail::equal_value(*first_value, kv.second))
                    throw std::runtime_error("inconsistent GGUF shard metadata: " + kv.first);
            }
            check_tensors(header, files.back().end);
            for (auto& tensor : header.tensors) m.tensors.push_back(std::move(tensor));
        }
        if (m.tensors.size() != uint64_t(split.tensors))
            throw std::runtime_error("GGUF split tensor count mismatch");
        // The assembled model can be written as one file; split keys describe its inputs only.
        m.kv.erase(std::remove_if(m.kv.begin(), m.kv.end(), [](const auto& kv) {
            return kv.first == "split.no" || kv.first == "split.count" || kv.first == "split.tensors.count";
        }), m.kv.end());
    }

    // Every file is mapped and its tensors read in place, a shard's data section placed after the one before in the model's offsets, so a model larger than host memory loads; the pages are touched once in the steps the progress reports, so the model is resident before its first pass and the host can still drop what a device copied.
    if (m.tensors.empty()) {
        if (progress) progress(0, 0);
        return m;
    }
    size_t payload = 0, base = 0;
    m.offsets.reserve(m.tensors.size());
    for (const auto& file : files) {
        // A shard of metadata alone may end before its data section would begin, and holds nothing to map.
        if (file.begin == file.end) continue;
        auto map = std::make_shared<const format::MappedFile>(file.path);
        const uint64_t start = file.data_start;
        if (start > map->size()) throw std::ios_base::failure("GGUF data section exceeds file extent");
        const size_t size = size_t(map->size() - start);
        for (size_t i = file.begin; i < file.end; ++i) {
            const auto& t = m.tensors[i];
            if (t.offset % alignof(float)) throw std::runtime_error("GGUF tensor offset is not float aligned");
            if (t.offset > size || t.data_size() > size - t.offset)
                throw std::ios_base::failure("GGUF tensor exceeds file extent");
            m.offsets.push_back(size_t(checked_add(base, t.offset)));
            payload = size_t(checked_add(payload, t.data_size()));
        }
        m.segments.push_back({map, size_t(start), base, size});
        base = size_t(aligned_size(checked_add(base, size), alignof(float)));
    }
    if (progress && payload) progress(0, payload);
    // Pages touched here stay resident only while the host can hold them: a payload larger than the memory available is evicted before a device copies it and read from disk twice, so it is left to be read once by whoever reads it.
    const auto available = core::host_memory_available();
    if (available && payload > *available) {
        if (progress) progress(payload, payload);
        return m;
    }
    size_t completed = 0;
    volatile uint8_t sink = 0;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const uint8_t* p = m.tensor_data(i);
        const size_t bytes = m.tensor_bytes(i);
        for (size_t offset = 0; offset < bytes;) {
            const size_t chunk = std::min(bytes - offset, size_t(8 * 1024 * 1024));
            uint8_t acc = 0;
            for (size_t b = 0; b < chunk; b += 4096) acc ^= p[offset + b];
            sink = sink ^ acc;
            offset += chunk;
            completed += chunk;
            if (progress && completed < payload) progress(completed, payload);
        }
    }
    if (progress) progress(completed, payload);
    return m;
}

// GGUF is the reference implementation of the format::ModelFormat interface.
// It wraps a GGUFModel (already read into memory) and exposes its tensors and metadata through the format-agnostic view.
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
        const gguf::MetaValue* v = m_.find(key);
        return v && v->vtype == gguf::V_STRING ? v->s : "";
    }

    uint64_t metadata_u64(const std::string& key) const override {
        const gguf::MetaValue* v = m_.find(key);
        if (!v) return 0;
        switch (v->vtype) {
            case gguf::V_UINT8: case gguf::V_UINT16: case gguf::V_UINT32: case gguf::V_UINT64: return v->u;
            case gguf::V_INT8: case gguf::V_INT16: case gguf::V_INT32: case gguf::V_INT64: return (uint64_t)v->i;
            default: return 0;
        }
    }

private:
    gguf::GGUFModel m_;
};

} // namespace gguf

// Auto-detect the format from the file header and open it.
// Currently only GGUF is implemented; the magic check is the extension point for future formats.
inline format::ModelFormatPtr format::open(const std::string& path, const LoadProgress& progress) {
    std::ifstream is(std::filesystem::u8path(path), std::ios::binary);
    if (!is) throw std::runtime_error("cannot open file: " + path);
    uint32_t magic = 0;
    is.read((char*)&magic, 4);
    if (is && magic == gguf::MAGIC) return std::make_shared<gguf::GGUFFormat>(gguf::read_gguf(path, progress));
    return nullptr;
}
