#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backends/backend.hpp"
#include "core/host_memory.hpp"
#include "core/job_threads.hpp"
#include "format/file_reader.hpp"
#include "quant/types.hpp"

namespace infer {

// Bounded immutable access to rows stored in their original GGUF format.
// A file-backed table's rows are read on kReadThreads at once, its readers the table's, and a read can be started ahead of its use (start) and taken when needed (finish).
class WeightRows {
public:
    // A cold row of a compressed file system costs a record's read and decompression, so a table's rows are read in parallel (docs/STATUS.md, Lookup rows read ahead).
    static constexpr size_t kReadThreads = 16;

    struct Rows {
        // These views are borrowed from Scratch until its next read.
        const uint8_t* data = nullptr;
        const uint32_t* ids = nullptr;
        const size_t* remap = nullptr;
        size_t unique = 0, requested = 0, row_bytes = 0;
    };

    class Scratch {
    public:
        Scratch(const WeightRows& source, size_t max_rows)
            : type_(source.type_), columns_(source.columns_), row_bytes_(source.row_bytes_),
              max_rows_(max_rows), file_(bool(source.reader_)),
              raw_(backend::size_mul(max_rows, source.row_bytes_)),
              order_(max_rows), remap_(max_rows), unique_(max_rows) {}
        Scratch(const Scratch&) = delete;
        Scratch& operator=(const Scratch&) = delete;
        Scratch(Scratch&&) = default;
        Scratch& operator=(Scratch&&) = default;
        // A read in flight writes only memory this scratch owns, so it is waited for before that memory goes.
        ~Scratch() {
            if (pending_.valid()) pending_.wait();
        }

    private:
        friend class WeightRows;
        uint32_t type_;
        size_t columns_, row_bytes_, max_rows_;
        bool file_;
        core::HostPages raw_;
        std::vector<size_t> order_, remap_;
        std::vector<uint32_t> unique_;
        size_t unique_count_ = 0, requested_ = 0;
        bool started_ = false;
        std::future<void> pending_;          // a file read started and not yet finished
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
        readers_ = std::make_shared<Readers>();
        if (reader_->direct())
            for (size_t t = 0; t < kReadThreads; ++t) readers_->windows.emplace_back(reader_->granule());
    }

    uint32_t type() const { return type_; }
    size_t columns() const { return columns_; }
    size_t rows() const { return rows_; }
    size_t row_bytes() const { return row_bytes_; }
    size_t bytes() const { return bytes_; }
    // The host bytes of the table's I/O windows, which the fit counts once for the table: a granule for each reader of a direct source, none for a buffered one, whose reads land in the rows.
    size_t staging_bytes() const { return reader_ && reader_->direct() ? backend::size_mul(kReadThreads, reader_->granule()) : 0; }

    // Reads `ids` into `scratch`: from a file on its threads, returning at once, and from memory before it returns; finish takes the rows.
    // A read the caller abandoned is waited for and dropped first, since a scratch holds one at a time; the ids are not kept, a request the table refuses throws here and a failed file read at finish.
    void start(const uint32_t* ids, size_t count, Scratch& scratch) const {
        if (scratch.pending_.valid()) {
            scratch.pending_.wait();
            scratch.pending_ = {};
        }
        scratch.started_ = false;
        if (scratch.type_ != type_ || scratch.columns_ != columns_ || scratch.row_bytes_ != row_bytes_)
            throw std::invalid_argument("weight rows: incompatible scratch");
        if (scratch.file_ != bool(reader_)) throw std::invalid_argument("weight rows: scratch was made for another kind of source");
        if (count > scratch.max_rows_)
            throw std::length_error("weight rows: request exceeds scratch capacity");
        if (count && !ids) throw std::invalid_argument("weight rows: nonempty request has no IDs");
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= rows_) throw std::out_of_range("weight rows: row id is outside the table");
        scratch.unique_count_ = 0;
        scratch.requested_ = count;
        // A read counts as started only once it is under way, so a start that fails leaves finish refusing rather than giving the last read's rows.
        if (!count) {
            scratch.started_ = true;
            return;
        }

        for (size_t i = 0; i < count; ++i) scratch.order_[i] = i;
        std::sort(scratch.order_.begin(), scratch.order_.begin() + count,
                  [&](size_t a, size_t b) { return ids[a] < ids[b]; });
        for (size_t i = 0; i < count; ++i) {
            const uint32_t id = ids[scratch.order_[i]];
            if (!scratch.unique_count_ || scratch.unique_[scratch.unique_count_ - 1] != id)
                scratch.unique_[scratch.unique_count_++] = id;
            scratch.remap_[scratch.order_[i]] = scratch.unique_count_ - 1;
        }

        if (!reader_) {
            read_memory(scratch);
            scratch.started_ = true;
            return;
        }
        // Contiguous ids are one span and each span one read, the spans dealt to the table's readers, the thread std::async starts among them; the table reads for one scratch at a time.
        std::vector<Span> spans;
        for (size_t i = 0, out = 0; i < scratch.unique_count_;) {
            size_t end = i + 1;
            while (end < scratch.unique_count_ && scratch.unique_[end] == scratch.unique_[end - 1] + 1u) ++end;
            spans.push_back({offset_ + backend::size_mul(scratch.unique_[i], row_bytes_), scratch.raw_.data() + backend::size_mul(out, row_bytes_),
                             backend::size_mul(end - i, row_bytes_)});
            out += end - i;
            i = end;
        }
        scratch.pending_ = std::async(std::launch::async, [readers = readers_, spans = std::move(spans), reader = reader_] {
            std::lock_guard<std::mutex> lk(readers->m);
            if (!readers->threads) readers->threads = std::make_unique<core::JobThreads>(kReadThreads - 1);
            const size_t lanes = std::min(spans.size(), kReadThreads);
            readers->threads->run(lanes, [&](size_t lane) {
                for (size_t i = lane; i < spans.size(); i += lanes) {
                    if (reader->direct()) read_direct(*reader, spans[i], readers->windows[lane].data(), reader->granule());
                    else if (reader->read(spans[i].source, spans[i].destination, spans[i].bytes) != spans[i].bytes)
                        throw std::runtime_error("weight rows: short file read");
                }
            });
        });
        scratch.started_ = true;
    }

    // The rows of the read started last, once read, borrowed from `scratch` until its next start.
    Rows finish(Scratch& scratch) const {
        if (!scratch.started_) throw std::logic_error("weight rows: no read started");
        scratch.started_ = false;
        if (scratch.pending_.valid()) {
            try {
                scratch.pending_.get();
            } catch (...) {
                scratch.unique_count_ = 0;
                throw;
            }
        }
        if (!scratch.requested_) return {nullptr, nullptr, nullptr, 0, 0, row_bytes_};
        return {scratch.raw_.data(), scratch.unique_.data(), scratch.remap_.data(), scratch.unique_count_, scratch.requested_, row_bytes_};
    }

private:
    struct Span {
        uint64_t source;
        uint8_t* destination;
        size_t bytes;
    };

    // A file table's readers: kReadThreads - 1 threads, started at the table's first read, beside the thread that starts it, and for a direct source an aligned window of a granule each.
    struct Readers {
        std::mutex m;
        std::unique_ptr<core::JobThreads> threads;
        std::vector<core::HostPages> windows;
    };

    void read_memory(Scratch& scratch) const {
        for (size_t i = 0; i < scratch.unique_count_; ++i)
            std::memcpy(scratch.raw_.data() + backend::size_mul(i, row_bytes_), memory_ + backend::size_mul(scratch.unique_[i], row_bytes_), row_bytes_);
    }

    // A span read through an aligned window, a chunk the window holds at a time.
    static void read_direct(const format::FileReader& reader, const Span& s, uint8_t* window, size_t capacity) {
        const size_t g = reader.granule();
        size_t done = 0;
        while (done < s.bytes) {
            const uint64_t at = s.source + done;
            const uint64_t aligned = at / g * g;
            const size_t lead = size_t(at - aligned);
            const size_t room = capacity - lead;
            const size_t take = std::min(s.bytes - done, room);
            if (!take) throw std::logic_error("weight rows: I/O window cannot hold one aligned chunk");
            const size_t request = (lead + take + g - 1) / g * g;
            const size_t got = reader.read(aligned, window, request);
            if (got < lead || got - lead < take) throw std::runtime_error("weight rows: short file read");
            std::memcpy(s.destination + done, window + lead, take);
            done += take;
        }
    }

    uint32_t type_;
    size_t columns_, rows_, row_bytes_, bytes_ = 0;
    const uint8_t* memory_ = nullptr;
    std::shared_ptr<const void> memory_owner_;
    std::shared_ptr<const format::FileReader> reader_;
    std::shared_ptr<Readers> readers_;
    uint64_t offset_ = 0;
};

} // namespace infer
