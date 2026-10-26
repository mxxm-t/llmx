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

// Bounded immutable access to rows stored in their original GGUF format, read from the table's file on its own readers, started ahead of their use (start) and taken when needed (finish).
// Its hot rows can be held in host memory (hold), where a row is copied from and never read from the file (docs/src/model-weight_rows.md).
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
              max_rows_(max_rows),
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
        core::HostPages raw_;
        std::vector<size_t> order_, remap_;
        std::vector<uint32_t> unique_;
        size_t unique_count_ = 0, requested_ = 0;
        bool started_ = false;
        std::future<void> pending_;          // a file read started and not yet finished
    };

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

    // The host bytes a store of `slots` rows of `row_bytes` takes in all: its arena, its index at load 3/4 rounded up to a power of two, and a row id and a reference byte a slot, each in whole pages.
    static size_t store_bytes_for(size_t slots, size_t row_bytes) {
        if (!slots) return 0;
        const size_t page = core::page_size();
        const auto pages = [page](size_t b) { return backend::size_mul((b + page - 1) / page, page); };
        return backend::size_add(backend::size_add(pages(backend::size_mul(slots, row_bytes)), pages(backend::size_mul(Store::index_size(slots), sizeof(uint64_t)))),
                                 backend::size_add(pages(backend::size_mul(slots, sizeof(uint32_t))), pages(slots)));
    }
    // The most slots of `row_bytes` whose store takes at most `bytes` (store_bytes_for).
    static size_t store_slots_within(size_t bytes, size_t row_bytes) {
        size_t lo = 0, hi = bytes / (row_bytes + 1) + 1;
        while (lo + 1 < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            (store_bytes_for(mid, row_bytes) <= bytes ? lo : hi) = mid;
        }
        return lo;
    }

    // Holds this table's hot rows in at most `bytes` of host memory from now on, none for 0; called before the table's first read.
    void hold(size_t bytes) const {
        std::lock_guard<std::mutex> lk(readers_->m);
        const size_t slots = std::min(rows_, store_slots_within(bytes, row_bytes_));
        readers_->store = slots ? std::make_unique<Store>(slots, row_bytes_) : nullptr;
    }
    // The host bytes of the table's store (store_bytes_for).
    size_t store_bytes() const {
        std::lock_guard<std::mutex> lk(readers_->m);
        return readers_->store ? store_bytes_for(readers_->store->slots(), row_bytes_) : 0;
    }
    // The table's own host bytes, which the fit counts once for the table: its store, and its I/O windows, a granule for each reader of a direct source and none for a buffered one, whose reads land in the rows.
    size_t host_bytes() const {
        return backend::size_add(store_bytes(), reader_->direct() ? backend::size_mul(kReadThreads, reader_->granule()) : 0);
    }

    // Reads `ids` into `scratch` on the table's threads, returning at once; finish takes the rows.
    // A read the caller abandoned is waited for and dropped first, since a scratch holds one at a time; the ids are not kept, a request the table refuses throws here and a failed file read at finish.
    void start(const uint32_t* ids, size_t count, Scratch& scratch) const {
        if (scratch.pending_.valid()) {
            scratch.pending_.wait();
            scratch.pending_ = {};
        }
        scratch.started_ = false;
        if (scratch.type_ != type_ || scratch.columns_ != columns_ || scratch.row_bytes_ != row_bytes_)
            throw std::invalid_argument("weight rows: incompatible scratch");
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

        // The table reads for one scratch at a time, under its readers' mutex for the whole task: the store's lookups, the misses' file reads, then their publish, so no lookup meets a publish.
        // Contiguous missing ids are one span and each span one read, the spans dealt to the table's readers, the thread std::async starts among them.
        scratch.pending_ = std::async(std::launch::async, [readers = readers_, reader = reader_, offset = offset_, row_bytes = row_bytes_,
                                                           unique = scratch.unique_.data(), n = scratch.unique_count_, raw = scratch.raw_.data()] {
            std::lock_guard<std::mutex> lk(readers->m);
            Store* store = readers->store.get();
            std::vector<char> held(n, 0);
            if (store)
                for (size_t i = 0; i < n; ++i) {
                    const size_t slot = store->find(unique[i]);
                    if (slot == Store::npos) continue;
                    std::memcpy(raw + i * row_bytes, store->row(slot), row_bytes);
                    store->touch(slot);
                    held[i] = 1;
                }
            std::vector<Span> spans;
            for (size_t i = 0; i < n;) {
                if (held[i]) { ++i; continue; }
                size_t end = i + 1;
                while (end < n && !held[end] && unique[end] == unique[end - 1] + 1u) ++end;
                spans.push_back({offset + backend::size_mul(unique[i], row_bytes), raw + backend::size_mul(i, row_bytes), backend::size_mul(end - i, row_bytes)});
                i = end;
            }
            if (spans.empty()) return;
            if (!readers->threads) readers->threads = std::make_unique<core::JobThreads>(kReadThreads - 1);
            const size_t lanes = std::min(spans.size(), kReadThreads);
            readers->threads->run(lanes, [&](size_t lane) {
                for (size_t i = lane; i < spans.size(); i += lanes) {
                    if (reader->direct()) read_direct(*reader, spans[i], readers->windows[lane].data(), reader->granule());
                    else if (reader->read(spans[i].source, spans[i].destination, spans[i].bytes) != spans[i].bytes)
                        throw std::runtime_error("weight rows: short file read");
                }
            });
            // Only rows whose whole read succeeded reach the store.
            if (store)
                for (size_t i = 0; i < n; ++i)
                    if (!held[i]) store->put(unique[i], raw + i * row_bytes);
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

    // A table's hot rows: packed a slot each in an arena committed as slots fill, found through an open-addressing index of {row id, slot} at load 3/4 with linear probing and backward-shift deletion,
    // and given up by CLOCK, one reference byte a slot, which matched LRU's hit rate on the tables' row traces (docs/CACHE-PLAN.md, section 9).
    class Store {
    public:
        static constexpr size_t npos = SIZE_MAX;

        Store(size_t slots, size_t row_bytes)
            : slots_(slots), row_bytes_(row_bytes), arena_(core::HostPages::reserved(backend::size_mul(slots, row_bytes))),
              ids_(core::HostPages::reserved(backend::size_mul(slots, sizeof(uint32_t)))), refs_(core::HostPages::reserved(slots)),
              index_(index_size(slots), kEmpty), shift_(64 - log2(index_.size())) {}

        // Index entries for `slots`: a power of two past 4/3 of them, so a probe stays short.
        static size_t index_size(size_t slots) {
            const size_t want = backend::size_add(slots, slots / 3 + 1);
            size_t n = 1;
            while (n < want) n <<= 1;
            return n;
        }
        size_t slots() const { return slots_; }

        // The slot holding `row`, or npos.
        size_t find(uint32_t row) const {
            for (size_t i = home(row);; i = (i + 1) & (index_.size() - 1)) {
                const uint64_t e = index_[i];
                if (e == kEmpty) return npos;
                if (uint32_t(e) == row) return size_t(e >> 32);
            }
        }
        const uint8_t* row(size_t slot) const { return arena_.data() + slot * row_bytes_; }
        void touch(size_t slot) { refs_.data()[slot] = 1; }

        // Keeps `bytes` as `row`; once every slot is used, the first slot CLOCK finds unreferenced is given up for it.
        // A row comes in referenced, so a publish gives up older rows before its own.
        void put(uint32_t row, const uint8_t* bytes) {
            size_t slot;
            if (used_ < slots_) {
                slot = used_++;
                if (used_ > committed_) {
                    // Pages are given a chunk of slots at a time, so the store takes memory as it fills.
                    const size_t upto = std::min(slots_, std::max(used_, committed_ + kCommitSlots));
                    arena_.commit(committed_ * row_bytes_, (upto - committed_) * row_bytes_);
                    ids_.commit(committed_ * sizeof(uint32_t), (upto - committed_) * sizeof(uint32_t));
                    refs_.commit(committed_, upto - committed_);
                    committed_ = upto;
                }
            } else {
                uint8_t* refs = refs_.data();
                while (refs[hand_]) {
                    refs[hand_] = 0;
                    hand_ = (hand_ + 1) % slots_;
                }
                slot = hand_;
                hand_ = (hand_ + 1) % slots_;
                erase(slot_row(slot));
            }
            std::memcpy(arena_.data() + slot * row_bytes_, bytes, row_bytes_);
            reinterpret_cast<uint32_t*>(ids_.data())[slot] = row;
            refs_.data()[slot] = 1;
            size_t i = home(row);
            while (index_[i] != kEmpty) i = (i + 1) & (index_.size() - 1);
            index_[i] = (uint64_t(slot) << 32) | row;
        }

    private:
        static constexpr uint64_t kEmpty = ~uint64_t(0);
        static constexpr size_t kCommitSlots = size_t(1) << 14;
        static unsigned log2(size_t n) {
            unsigned b = 0;
            while ((size_t(1) << b) < n) ++b;
            return b;
        }
        size_t home(uint32_t row) const { return shift_ >= 64 ? 0 : size_t((uint64_t(row) * 0x9E3779B97F4A7C15ull) >> shift_); }
        uint32_t slot_row(size_t slot) const { return reinterpret_cast<const uint32_t*>(ids_.data())[slot]; }

        // Takes `row` out of the index, moving each later entry of its run back into the gap where its home allows.
        void erase(uint32_t row) {
            const size_t mask = index_.size() - 1;
            size_t gap = home(row);
            while (uint32_t(index_[gap]) != row) gap = (gap + 1) & mask;
            for (size_t j = (gap + 1) & mask;; j = (j + 1) & mask) {
                const uint64_t e = index_[j];
                if (e == kEmpty) break;
                const size_t h = home(uint32_t(e));
                // The entry may fill the gap where the gap lies cyclically between its home and its place.
                if (((j - h) & mask) >= ((j - gap) & mask)) {
                    index_[gap] = e;
                    gap = j;
                }
            }
            index_[gap] = kEmpty;
        }

        size_t slots_, row_bytes_, used_ = 0, committed_ = 0, hand_ = 0;
        core::HostPages arena_, ids_, refs_;
        std::vector<uint64_t> index_;
        unsigned shift_;
    };

    // A file table's readers: kReadThreads - 1 threads, started at the table's first read, beside the thread that starts it, for a direct source an aligned window of a granule each, and its store of hot rows.
    struct Readers {
        std::mutex m;
        std::unique_ptr<core::JobThreads> threads;
        std::vector<core::HostPages> windows;
        std::unique_ptr<Store> store;
    };

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
    std::shared_ptr<const format::FileReader> reader_;
    std::shared_ptr<Readers> readers_;
    uint64_t offset_ = 0;
};

} // namespace infer
