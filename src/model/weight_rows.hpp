#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backends/backend.hpp"
#include "core/host_memory.hpp"
#include "format/file_reader.hpp"
#include "quant/types.hpp"

namespace infer {

// Bounded immutable access to rows stored in their original GGUF format.
class WeightRows {
public:
    struct Rows {
        // These views are borrowed from Scratch until its next read.
        const uint8_t* data = nullptr;
        const uint32_t* ids = nullptr;
        const size_t* remap = nullptr;
        size_t unique = 0, requested = 0, row_bytes = 0;
    };

    class Scratch {
    public:
        Scratch(const WeightRows& source, size_t max_rows, size_t io_bytes)
            : type_(source.type_), columns_(source.columns_), row_bytes_(source.row_bytes_),
              max_rows_(max_rows), file_(bool(source.reader_)),
              direct_(source.reader_ && source.reader_->direct()),
              granule_(source.reader_ ? source.reader_->granule() : 0),
              raw_(backend::size_mul(max_rows, source.row_bytes_)),
              order_(max_rows), remap_(max_rows), unique_(max_rows) {
            if (source.reader_) {
                const size_t granule = source.reader_->granule();
                if (!io_bytes || (source.reader_->direct() && io_bytes < granule))
                    throw std::invalid_argument("weight rows: I/O window is smaller than its alignment");
                if (source.reader_->direct()) io_bytes = io_bytes / granule * granule;
                io_capacity_ = io_bytes;
                io_ = core::HostPages(io_bytes);
            }
        }
        Scratch(const Scratch&) = delete;
        Scratch& operator=(const Scratch&) = delete;
        Scratch(Scratch&&) = default;
        Scratch& operator=(Scratch&&) = default;

    private:
        friend class WeightRows;
        uint32_t type_;
        size_t columns_, row_bytes_, max_rows_;
        bool file_, direct_;
        size_t granule_, io_capacity_ = 0;
        core::HostPages raw_, io_;
        std::vector<size_t> order_, remap_;
        std::vector<uint32_t> unique_;
        size_t unique_count_ = 0;
    };

    WeightRows(uint32_t type, size_t columns, size_t rows, const uint8_t* memory, size_t bytes,
               std::shared_ptr<const void> memory_owner = {})
        : type_(type), columns_(columns), rows_(rows), row_bytes_(quant::row_bytes(type, columns)) {
        if (!columns || (rows && rows - 1 > size_t(UINT32_MAX)))
            throw std::invalid_argument("weight rows: dimensions exceed row ID range");
        const size_t expected = backend::size_mul(row_bytes_, rows);
        if (bytes != expected || (bytes && !memory))
            throw std::invalid_argument("weight rows: memory extent does not match its shape");
        memory_ = memory;
        memory_owner_ = std::move(memory_owner);
        bytes_ = bytes;
    }

    WeightRows(uint32_t type, size_t columns, size_t rows,
               std::shared_ptr<const format::FileReader> reader, uint64_t offset, size_t bytes)
        : type_(type), columns_(columns), rows_(rows), row_bytes_(quant::row_bytes(type, columns)),
          reader_(std::move(reader)), offset_(offset) {
        if (!columns || (rows && rows - 1 > size_t(UINT32_MAX)))
            throw std::invalid_argument("weight rows: dimensions exceed row ID range");
        const size_t expected = backend::size_mul(row_bytes_, rows);
        if (!reader_) throw std::invalid_argument("weight rows: no file reader");
        if (bytes != expected || offset > reader_->size() || bytes > reader_->size() - offset)
            throw std::invalid_argument("weight rows: file extent does not match its shape");
        bytes_ = bytes;
    }

    uint32_t type() const { return type_; }
    size_t columns() const { return columns_; }
    size_t rows() const { return rows_; }
    size_t row_bytes() const { return row_bytes_; }
    size_t bytes() const { return bytes_; }
    size_t io_bytes() const { return reader_ ? reader_->granule() : 0; }

    Rows read(const uint32_t* ids, size_t count, Scratch& scratch) const {
        if (scratch.type_ != type_ || scratch.columns_ != columns_ || scratch.row_bytes_ != row_bytes_)
            throw std::invalid_argument("weight rows: incompatible scratch");
        const bool file = bool(reader_), direct = reader_ && reader_->direct();
        const size_t granule = reader_ ? reader_->granule() : 0;
        if (scratch.file_ != file || scratch.direct_ != direct || scratch.granule_ != granule)
            throw std::invalid_argument("weight rows: scratch has incompatible I/O alignment");
        if (count > scratch.max_rows_)
            throw std::length_error("weight rows: request exceeds scratch capacity");
        if (count && !ids) throw std::invalid_argument("weight rows: nonempty request has no IDs");
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= rows_) throw std::out_of_range("weight rows: row id is outside the table");
        scratch.unique_count_ = 0;
        if (!count) return {nullptr, nullptr, nullptr, 0, 0, row_bytes_};

        for (size_t i = 0; i < count; ++i) scratch.order_[i] = i;
        std::sort(scratch.order_.begin(), scratch.order_.begin() + count,
                  [&](size_t a, size_t b) { return ids[a] < ids[b]; });
        for (size_t i = 0; i < count; ++i) {
            const uint32_t id = ids[scratch.order_[i]];
            if (!scratch.unique_count_ || scratch.unique_[scratch.unique_count_ - 1] != id)
                scratch.unique_[scratch.unique_count_++] = id;
            scratch.remap_[scratch.order_[i]] = scratch.unique_count_ - 1;
        }

        try {
            read_unique(scratch);
        } catch (...) {
            scratch.unique_count_ = 0;
            throw;
        }
        return {scratch.raw_.data(), scratch.unique_.data(), scratch.remap_.data(),
                scratch.unique_count_, count, row_bytes_};
    }

private:
    void read_unique(Scratch& scratch) const {
        uint8_t* output = scratch.raw_.data();
        size_t out_row = 0, i = 0;
        while (i < scratch.unique_count_) {
            size_t end = i + 1;
            while (end < scratch.unique_count_ && scratch.unique_[end] == scratch.unique_[end - 1] + 1u) ++end;
            const size_t first = scratch.unique_[i];
            const size_t nrows = end - i;
            const size_t bytes = backend::size_mul(nrows, row_bytes_);
            const uint64_t source = offset_ + backend::size_mul(first, row_bytes_);
            uint8_t* destination = output + backend::size_mul(out_row, row_bytes_);
            if (reader_) {
                if (reader_->direct()) read_direct(source, destination, bytes, scratch);
                else read_buffered(source, destination, bytes, scratch);
            } else {
                std::memcpy(destination, memory_ + backend::size_mul(first, row_bytes_), bytes);
            }
            out_row += nrows;
            i = end;
        }
    }

    void read_buffered(uint64_t source, uint8_t* destination, size_t bytes, Scratch& scratch) const {
        size_t done = 0;
        while (done < bytes) {
            const size_t take = std::min(bytes - done, scratch.io_capacity_);
            const size_t got = reader_->read(source + done, scratch.io_.data(), take);
            if (got != take) throw std::runtime_error("weight rows: short file read");
            std::memcpy(destination + done, scratch.io_.data(), take);
            done += take;
        }
    }

    void read_direct(uint64_t source, uint8_t* destination, size_t bytes, Scratch& scratch) const {
        const size_t g = reader_->granule(), capacity = scratch.io_capacity_;
        size_t done = 0;
        while (done < bytes) {
            const uint64_t at = source + done;
            const uint64_t aligned = at / g * g;
            const size_t lead = size_t(at - aligned);
            const size_t room = capacity - lead;
            const size_t take = std::min(bytes - done, room);
            if (!take) throw std::logic_error("weight rows: I/O window cannot hold one aligned chunk");
            const size_t request = (lead + take + g - 1) / g * g;
            const size_t got = reader_->read(aligned, scratch.io_.data(), request);
            if (got < lead || got - lead < take) throw std::runtime_error("weight rows: short file read");
            std::memcpy(destination + done, scratch.io_.data() + lead, take);
            done += take;
        }
    }

    uint32_t type_;
    size_t columns_, rows_, row_bytes_, bytes_ = 0;
    const uint8_t* memory_ = nullptr;
    std::shared_ptr<const void> memory_owner_;
    std::shared_ptr<const format::FileReader> reader_;
    uint64_t offset_ = 0;
};

} // namespace infer
