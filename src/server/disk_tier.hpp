#pragma once
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif
#include "config.hpp"
#include "core/sha.hpp"
#include "format/file_digest.hpp"
#include "format/gguf.hpp"
#include "inference/read_ring.hpp"
#include "model/runtime.hpp"
#include "server/disk_index.hpp"
#include "server/disk_store.hpp"

// The disk tier's moving parts below the scheduler (docs/DISK-TIER.md): the store, made once the model file's digest is known, and the one write in flight.
// What is written, what room drops and the index of what is on disk are the scheduler's.

namespace server {

// What the build and the host give an entry's identity (docs/DISK-TIER.md, The entry file).
struct BuildFacts {
    std::string revision;   // the Git revision built from, which the identity leaves out
    std::string numerics;   // the fingerprint of the sources that decide a history's bits (cmake/numerics-sources.txt)
    std::string compiler;   // the compiler's own version text
    std::string flags;      // the compiler, configuration, flags and options the build system used
    std::string shaders;    // the shader compiler's version
    std::string libm;       // the C library the math functions come from, where it names itself
};
inline BuildFacts build_facts() {
    BuildFacts b;
    b.revision = LLMX_BUILD_REVISION;
    b.numerics = LLMX_NUMERICS;
    b.flags = LLMX_BUILD_FLAGS;
    b.shaders = LLMX_GLSLC_VERSION;
    b.libm = "unrecorded";
#if defined(_MSC_VER)
    b.compiler = "msvc " + std::to_string(_MSC_FULL_VER);
#else
    b.compiler = __VERSION__;
#endif
#if defined(__GLIBC__)
    b.libm = std::string("glibc ") + gnu_get_libc_version();
#endif
    return b;
}

// The text an entry's identity is the digest of, a component a line: the model file's digest, what can change a history's bits in the build, and the model's host layout (Model::host_identity), which names each device and its driver.
// The revision is not among them, so a build that changes none of them reads the entries of the one before.
inline std::string disk_identity(const std::string& digest, const std::string& layout, const BuildFacts& build) {
    std::string flat = layout;
    for (char& c : flat)
        if (c == '\n') c = '|';
    return "model: " + digest + "\nnumerics: " + build.numerics + "\ncompiler: " + build.compiler + "\nflags: " + build.flags + "\nshaders: " + build.shaders + "\nlibm: " + build.libm + "\nentries: " + std::to_string(DiskStore::kVersion) + "\nlayout: " + flat + "\n";
}

// Every file of the model at `path` whose bytes its weights are: the path, and each shard of a set the reader finds tensors in.
// A first shard of metadata alone is the same bytes for models whose tensors differ, so the identity takes them all.
inline std::vector<std::string> model_files(const std::string& path) {
    std::vector<std::string> files{path};
    for (const auto& segment : gguf::read_gguf(path).segments)
        if (std::find(files.begin(), files.end(), segment.path) == files.end()) files.push_back(segment.path);
    return files;
}

// The server's disk tier, from --disk-cache-bytes, --disk-cache-dir, --disk-cache-floor, --disk-cache-keep and --disk-cache-max-age; `bytes` 0 keeps none.
// How long a server under --disk-cache-keep has had nothing to do before it writes ahead what a stop would have to, so a turn that follows at once meets no copy off the devices.
constexpr std::chrono::seconds kDiskIdle{5};

struct DiskOptions {
    uint64_t bytes = 0;
    std::string dir;
    std::optional<uint64_t> floor;   // none takes the larger of 16 GiB and a twentieth of the file system (check_disk_cache)
    bool keep = false;               // at a clean exit flush memory to disk and leave the entries for the next server, which adopts them
    uint64_t max_age = 24 * 3600;    // seconds an entry may go unused before it is deleted, 0 for no limit
    std::string model_path;          // the model file, or the first shard of a set, whose files' digests every entry's identity carries (model_files)
    std::chrono::milliseconds pace{0};   // DiskStore::Options::pace, for tests
    std::optional<BuildFacts> build;     // the build's facts in place of build_facts(), for tests
    bool adopt_proved = false;           // adopt another build's entries unproved and prove each at its first use (--disk-cache-adopt proved)
    std::chrono::milliseconds idle{kDiskIdle};   // how long nothing runs before the idle writes start, shorter in tests
    std::optional<std::chrono::milliseconds> wait;   // the bound of a wait for a restore or a save, kWaitBound where none is given, shorter in tests
    std::optional<std::chrono::milliseconds> read_wait;   // a read's bound in place of the time its rows take to compute, for tests a slow runner must not time out
    std::optional<double> recompute_row_ms;   // a row's time to compute in place of the measured prompt rate where it caps a wait for a save, for tests
    std::function<bool(const char* step)> fault;   // DiskStore::Options::fault, for tests
};

// Before the server listens: makes the directory, takes the default floor, prints the directory, the file system's free space and the floor, and refuses a tier the host tier does not feed or whose cap and floor together exceed the free space.
inline void check_disk_cache(DiskOptions& o, size_t host_cap) {
    if (!o.bytes) return;
    if (!host_cap) throw std::runtime_error("server: the disk cache keeps what the host cache drops, and the host cache holds none (--host-cache-bytes 0, or every cache on the CPU)");
    const auto root = std::filesystem::u8path(o.dir);
    std::filesystem::create_directories(root);
    const auto space = std::filesystem::space(root);
    if (!o.floor) o.floor = std::max<uint64_t>(uint64_t(16) << 30, (uint64_t)space.capacity / 20);
    std::fprintf(stderr, "server: disk cache of %llu MiB in %s, %llu MiB free, keeping %llu MiB free\n", (unsigned long long)(o.bytes >> 20), root.u8string().c_str(),
                 (unsigned long long)(space.available >> 20), (unsigned long long)(*o.floor >> 20));
    if (o.bytes > space.available || *o.floor > space.available - o.bytes)
        throw std::runtime_error("server: a disk cache of " + std::to_string(o.bytes) + " bytes and a floor of " + std::to_string(*o.floor) + " bytes need more than the " +
                                 std::to_string(space.available) + " bytes free in " + root.u8string());
}

// How long a clean exit under --disk-cache-keep spends writing what memory holds to disk (docs/DISK-TIER.md, Keeping entries across a restart).
constexpr std::chrono::seconds kDiskFlush{20};

// A history read from disk straight into its device storage through the read ring, with no whole copy in host memory, stepped by the scheduler (docs/DISK-TIER.md, Restore).
// The store owns the files' format and checks, the model where each byte goes (Model::restore_into); a failure leaves the copies made charged until they retire.
class StreamedRead {
public:
    enum class State { running, done, failed };
    // One file of the path: its key, the layout its header must carry, and its payload as segments of a device's image, `skip` for the bytes a path does not use.
    struct File {
        uint64_t key = 0;
        std::vector<size_t> layout;
        struct Segment {
            size_t device = 0, image = 0, bytes = 0;
            bool skip = false;
        };
        std::vector<Segment> segments;
        uint64_t payload = 0;
    };

    StreamedRead(const DiskStore& store, std::vector<File> files, infer::RestoreInto& into, std::function<void()> wake)
        : store_(store), files_(std::move(files)), into_(into), wake_(std::move(wake)), opened_(files_.size()) {
        for (size_t f = 0; f < files_.size(); ++f)
            for (uint64_t c = 0; c * DiskStore::kChunk < files_[f].payload; ++c) chunks_.push_back({f, c});
        // Each device's image offsets where its runs start, so an image byte finds its run.
        starts_.resize(into_.runs.size());
        for (size_t d = 0; d < into_.runs.size(); ++d) {
            size_t at = 0;
            for (const infer::RestoreInto::Run& r : into_.runs[d]) {
                starts_[d].push_back(at);
                at += r.bytes;
            }
        }
        ring_ = std::make_unique<infer::ReadRing>(chunks_.size(), DiskStore::kChunk, [this](size_t k, uint8_t* slot) { return fill(k, slot); });
        // Each device's view of every slot where it copies out of host pages in place; a write where it does not.
        views_.resize(into_.backends.size());
        for (size_t d = 0; d < into_.backends.size(); ++d)
            for (size_t i = 0; i < ring_->slots(); ++i) {
                backend::BufferPtr v = into_.backends[d]->wrap_host(ring_->pages(i).data(), ring_->pages(i).size());
                if (!v) {
                    views_[d].clear();
                    break;
                }
                views_[d].push_back(std::move(v));
            }
        last_.assign(into_.backends.size(), 0);
        // The readers start only now: a view that throws above leaves no reader to join, one of which could be waking the scheduler whose lock the caller holds.
        ring_->start();
    }
    StreamedRead(const StreamedRead&) = delete;
    StreamedRead& operator=(const StreamedRead&) = delete;

    // The slots whose copies have retired given back, and the read's state; it takes no hold and never waits, since Backend::done records nothing and a device backend's does not wait.
    State poll() {
        if (state_ != State::running) return state_;
        retire();
        if (taken_ == chunks_.size() && pending_.empty()) state_ = State::done;
        return state_;
    }
    // Whether a full slot waits to be sent, or a reader's failure to be taken, either of which needs send().
    bool ready() const { return state_ == State::running && ring_->ready(); }
    // Every full slot's bytes copied to where they go and submitted, under the caller's hold of the devices; it never waits but on a failed submission.
    void send() {
        if (state_ != State::running) return;
        try {
            while (const std::optional<infer::ReadRing::Full> full = ring_->next(false)) send(*full);
        } catch (const Failure& e) {
            failed_key_ = e.key;
            failed(e.what());
        } catch (const std::exception& e) {
            failed(e.what());
        }
    }
    // Asks the readers to stop: a cancel, or a timeout at the read's bound; what is in flight stays charged until idle().
    void cancel() {
        ring_->stop();
        if (state_ == State::running) state_ = State::failed;
    }
    // Whether the readers have ended and every copy has retired, so the ring, its pages and the history's blocks can go with no wait.
    bool idle() {
        retire();
        return ring_->idle() && pending_.empty();
    }
    // Where a failure came from: the file whose read failed, 0 for a cancel, and why.
    uint64_t failed_key() const { return failed_key_; }
    const std::string& why() const { return why_; }
    // Each device's ticket of its last copy, which the restored history's passes go behind (Model::restored).
    const std::vector<backend::Ticket>& last() const { return last_; }
    uint64_t bytes() const { return bytes_; }

private:
    struct Chunk {
        size_t file;
        uint64_t index;
    };
    struct Pending {
        size_t piece;
        std::vector<backend::Ticket> tickets;
    };

    // On a reader thread: chunk k's file opened once, its header checked, and the chunk read and checked against its CRC.
    size_t fill(size_t k, uint8_t* slot) {
        const Chunk& c = chunks_[k];
        const DiskStore::Opened* o = nullptr;
        {
            std::lock_guard<std::mutex> lk(open_);
            if (!opened_[c.file]) {
                std::string why;
                std::optional<DiskStore::Opened> got = store_.open_entry(files_[c.file].key, files_[c.file].layout, why);
                if (!got) throw Failure{files_[c.file].key, why};
                opened_[c.file] = std::make_unique<DiskStore::Opened>(std::move(*got));
            }
            o = opened_[c.file].get();
        }
        size_t n = 0;
        try {
            n = DiskStore::read_chunk(*o, c.index, slot);
        } catch (const std::exception& e) {
            throw Failure{files_[c.file].key, e.what()};
        }
        if (wake_) wake_();
        return n;
    }
    struct Failure : std::runtime_error {
        uint64_t key;
        Failure(uint64_t k, const std::string& why) : std::runtime_error(why), key(k) {}
    };
    void failed(const char* why) {
        why_ = why;
        ring_->stop();
        state_ = State::failed;
    }

    // A full slot's chunk sent: each used segment it holds mapped through its device's runs, a copy out of the slot where the device reads it in place and a write otherwise.
    void send(const infer::ReadRing::Full& full) {
        const Chunk& c = chunks_[full.piece];
        const File& f = files_[c.file];
        const uint64_t from = c.index * DiskStore::kChunk, to = from + full.bytes;
        const uint8_t* data = ring_->pages(full.slot).data();
        std::vector<char> used(into_.backends.size(), 0);
        Pending p{full.piece, std::vector<backend::Ticket>(into_.backends.size(), 0)};
        try {
            uint64_t at = 0;
            for (const File::Segment& s : f.segments) {
                const uint64_t lo = std::max(at, from), hi = std::min(at + s.bytes, to);
                if (!s.skip && lo < hi) {
                    used[s.device] = 1;
                    put(s.device, s.image + (size_t)(lo - at), data + (lo - from), full.slot, (size_t)(lo - from), (size_t)(hi - lo));
                }
                at += s.bytes;
            }
            for (size_t d = 0; d < used.size(); ++d)
                if (used[d]) p.tickets[d] = last_[d] = into_.backends[d]->submit();
        } catch (...) {
            // The chunk's copies retire before the slot they read can go: a device that submitted waits on its ticket, one that only recorded submits and waits, under the caller's hold.
            for (size_t d = 0; d < used.size(); ++d) {
                if (!used[d]) continue;
                backend::Backend& b = *into_.backends[d];
                try {
                    b.wait(p.tickets[d] ? p.tickets[d] : b.submit());
                } catch (...) {
                    // This sync assumes no other thread records on the device, which the caller's hold of the stages gives (the server's quieted()).
                    b.sync();
                }
            }
            throw;
        }
        pending_.push_back(std::move(p));
        bytes_ += full.bytes;
        ++taken_;
    }
    // `n` bytes of device d's image from `image` on, which are at `in_slot` of the slot, to the runs they fall in.
    void put(size_t d, size_t image, const uint8_t* src, size_t slot, size_t in_slot, size_t n) {
        const std::vector<infer::RestoreInto::Run>& runs = into_.runs[d];
        size_t r = (size_t)(std::upper_bound(starts_[d].begin(), starts_[d].end(), image) - starts_[d].begin()) - 1;
        while (n) {
            const infer::RestoreInto::Run& run = runs[r];
            const size_t off = image - starts_[d][r], k = std::min(n, run.bytes - off);
            backend::Backend& b = *into_.backends[d];
            if (run.host) std::memcpy(run.host + run.offset + off, src, k);
            else if (!views_[d].empty()) b.copy(run.target(), run.offset + off, *views_[d][slot], in_slot, k);
            else b.write(run.target(), run.offset + off, src, k);
            image += k;
            src += k;
            in_slot += k;
            n -= k;
            ++r;
        }
    }
    // The slots whose copies have retired given back, in order.
    void retire() {
        while (!pending_.empty()) {
            const Pending& p = pending_.front();
            for (size_t d = 0; d < p.tickets.size(); ++d)
                if (p.tickets[d] && !into_.backends[d]->done(p.tickets[d])) return;
            ring_->free(p.piece);
            pending_.erase(pending_.begin());
        }
    }

    const DiskStore& store_;
    std::vector<File> files_;
    infer::RestoreInto& into_;
    const std::function<void()> wake_;
    std::vector<Chunk> chunks_;
    std::vector<std::vector<size_t>> starts_;
    std::mutex open_;
    std::vector<std::unique_ptr<DiskStore::Opened>> opened_;
    std::vector<backend::Ticket> last_;
    std::vector<Pending> pending_;
    size_t taken_ = 0;
    uint64_t bytes_ = 0, failed_key_ = 0;
    std::string why_;
    State state_ = State::running;
    std::unique_ptr<infer::ReadRing> ring_;            // after what its readers read, so they stop first
    std::vector<std::vector<backend::BufferPtr>> views_;   // after the ring, so they go before its pages
};

class DiskTier {
public:
    // A call the store has finished: its key, whether it was a read, whether it succeeded, and why not.
    struct Finished {
        uint64_t key = 0;
        bool read = false, ok = false;
        std::string error;
        uint64_t ticket = 0;   // a read's own number, since two reads may be on one file
    };

    // Makes the store on a thread of its own, once the digest of each of the model's files is read, under the identity of those digests, the build and the model's host layout; nothing is written before that.
    // `wake` runs, under no lock of the tier's, when the store is made or refused and when a write finishes.
    DiskTier(const DiskOptions& options, const infer::Model& model, std::function<void()> wake) : options_(options), wake_(std::move(wake)) {
        const std::string layout = model.host_identity();
        starter_ = std::thread([this, layout] {
            std::unique_ptr<DiskStore> store;
            try {
                std::string digest;
                for (const std::string& file : model_files(options_.model_path))
                    digest += (digest.empty() ? "" : " ") + format::cached_file_sha256((std::filesystem::u8path(options_.dir) / "digests").u8string(), file);
                const std::string text = disk_identity(digest, layout, options_.build ? *options_.build : build_facts());
                const std::array<uint8_t, 32> identity = DiskStore::identity_of(text);
                DiskStore::Options o;
                o.root = options_.dir;
                o.floor = options_.floor.value_or(0);
                o.keep = options_.keep;
                o.max_age = options_.max_age;
                o.pace = options_.pace;
                o.fault = options_.fault;
                o.proved = options_.adopt_proved;
                o.identity = text;
                store = std::make_unique<DiskStore>(o, identity);
                std::fprintf(stderr, "server: disk cache in %s, writes %s the file cache, numerics %.16s, %zu entries adopted\n", store->directory().u8string().c_str(),
                             store->direct() ? "around" : "through", LLMX_NUMERICS, store->adopted().size());
                for (const std::string& why : store->refused()) std::fprintf(stderr, "server: a kept directory was not adopted, %s; it is kept until its age limit\n", why.c_str());
                refused_dirs_ = store->refused().size();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "server: no disk cache (%s)\n", e.what());
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                store_ = std::move(store);
                writing_ = store_ != nullptr;
                ready_ = true;
            }
            if (wake_) wake_();
        });
    }
    // Stops the write in flight; the store then leaves or removes its directory, and no call reads a slab any more.
    void close() {
        if (starter_.joinable()) starter_.join();
        std::unique_ptr<DiskStore> store;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (store_ && in_flight_) store_->cancel(in_flight_);
            store = std::move(store_);
        }
        store.reset();
    }
    ~DiskTier() { close(); }
    DiskTier(const DiskTier&) = delete;
    DiskTier& operator=(const DiskTier&) = delete;

    uint64_t cap() const { return options_.bytes; }
    bool keeps() const { return options_.keep; }
    std::chrono::milliseconds idle() const { return options_.idle; }
    uint64_t max_age() const { return options_.max_age; }

    // The entries the store adopted from servers that left them, once the store is made, and once.
    std::vector<DiskStore::Adopted> take_adopted() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<DiskStore::Adopted> out;
        if (store_ && !adopted_taken_) out = store_->adopted();
        adopted_taken_ = adopted_taken_ || store_ != nullptr;
        return out;
    }

    // Whether a write may start now: the store made, writing not stopped and no write in flight.
    // Writing stopped by a failure starts again once a check, a minute after the last, finds the floor and a tenth of the cap free.
    bool can_write() {
        std::lock_guard<std::mutex> lk(m_);
        if (store_ && !writing_ && std::chrono::steady_clock::now() - stopped_ >= std::chrono::minutes(1)) {
            stopped_ = std::chrono::steady_clock::now();
            const std::optional<uint64_t> free = store_->free_bytes();
            if (free && *free >= options_.floor.value_or(0) + options_.bytes / 10) writing_ = true;
        }
        return store_ && writing_ && !in_flight_;
    }

    // The store's measured write rate in bytes a second, 0 until it is made or where it has none.
    double write_rate() const {
        std::lock_guard<std::mutex> lk(m_);
        return store_ ? store_->write_rate() : 0.0;
    }

    // The write in flight, if it is a file of history `d` in blocks of `block` tokens: its payload bytes written and the seconds since it began; zeros otherwise.
    // The file's last block must be the history's too, the digests being a chain: histories that share a start, from the root below a file at row 0 up, share digests below the file's end.
    std::pair<uint64_t, double> writing_of(const std::vector<std::string>& d, size_t block) const {
        std::lock_guard<std::mutex> lk(m_);
        const size_t at = pending_.first / block, n = pending_.ends.size();
        if (!store_ || !key_ || at >= d.size() || d[at] != pending_.below) return {0, 0.0};
        if (n && (at + n >= d.size() || d[at + n] != pending_.ends.back())) return {0, 0.0};
        return store_->write_progress();
    }

    // Whether what servers before left is in the index (adopt has taken it), or the store's start failed.
    bool settled() const {
        std::lock_guard<std::mutex> lk(m_);
        return adopted_taken_ || (ready_ && !store_);
    }
    // Whether reads may start: the store made.
    bool readable() const {
        std::lock_guard<std::mutex> lk(m_);
        return store_ != nullptr;
    }

    // Stops write `key`, which finishes cancelled unless its file is already in place.
    void cancel(uint64_t key) {
        std::lock_guard<std::mutex> lk(m_);
        if (store_ && key == in_flight_) store_->cancel(key);
    }

    // Removes entry `key`'s file.
    void remove(uint64_t key) {
        std::lock_guard<std::mutex> lk(m_);
        if (store_) store_->evict(key);
    }

    // Records entry `key`'s last use.
    void touch(uint64_t key, std::filesystem::file_time_type used) {
        std::lock_guard<std::mutex> lk(m_);
        if (store_) store_->touch(key, used);
    }

    // The calls finished since the last call; a write that failed for any reason but a cancel stops writing, as a full or failing disk does (docs/DISK-TIER.md, Disk eviction and room).
    std::vector<Finished> finished() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<Finished> out;
        out.swap(done_);
        for (const Finished& f : out) {
            if (f.read) continue;   // counted a read, in settle
            in_flight_ = 0;
            if (f.ok) continue;
            if (f.error.find("cancel") != std::string::npos) continue;
            ++errors_;
            if (writing_) std::fprintf(stderr, "server: the disk cache stops writing (%s)\n", f.error.c_str());
            writing_ = false;
            stopped_ = std::chrono::steady_clock::now();
        }
        return out;
    }

    // Whether the tier still writes: false once a write failed or the store could not be made.
    bool writing() const {
        std::lock_guard<std::mutex> lk(m_);
        return !ready_ || writing_;
    }
    // The kept directories of another identity this server's start did not adopt (DiskStore::refused).
    size_t refused_dirs() const { return refused_dirs_; }
    size_t errors() const {
        std::lock_guard<std::mutex> lk(m_);
        return errors_;
    }


    // ---- What is on disk, and the one write and the reads in flight: the scheduler's thread only, under its lock (docs/DISK-TIER.md, Entries written as what changed). ----

    const DiskIndex& index() const { return index_; }
    // Whether the index gained or lost a file since this was last asked, so the scheduler looks again at which of its entries are on disk.
    bool take_changed() {
        const bool c = changed_;
        changed_ = false;
        return c;
    }
    // The tier's figures as the scheduler's stats give them: the files on disk and the states among them, their bytes, every finished write's bytes, and the files deleted while running to stay within the cap.
    // Why a file left the index: a file of this build covering it, a write nobody claimed, the cap, its age, a failed proof of its build, no whole path reaching it, or its read failing.
    enum class Gone { covered, unclaimed, cap, age, condemned, unreachable, read_failed, count };
    static const char* gone_name(Gone g) {
        static const char* const names[] = {"covered by this build's file", "an unclaimed write", "the cap", "its age", "its build's failed proof", "no whole path reaching it", "its read failing"};
        return names[(size_t)g];
    }
    struct Figures {
        std::array<size_t, (size_t)Gone::count> gone{};   // files that left the index since start, by reason
        uint64_t whole = 0, failed = 0;                   // writes published in the index, and writes that failed
        size_t files = 0, states = 0, capped = 0;
        uint64_t bytes = 0, written = 0;
        size_t unproved = 0, proved = 0, disproved = 0;   // another build's files not proved yet, and since the start those proofs made this build's and those deleted with a build that failed one
    };
    Figures figures() const { return {gone_, whole_, failed_, index_.files().size(), index_.states(), capped_, index_.bytes(), written_, unproved(), proved_, disproved_}; }
    // Once an idle server's writes have ended: one line of what went to disk since the last such line.
    void say_written() {
        if (written_ == said_) return;
        std::fprintf(stderr, "server: %.1f MiB written to disk since the idle writes last ended; %zu files on disk, %zu of them states\n", (double)(written_ - said_) / (1 << 20), index_.files().size(), index_.states());
        said_ = written_;
    }
    size_t calls() const { return (key_ ? 1 : 0) + reads_.size() + streams_.size(); }
    // The host memory the tier's own calls hold beyond the host tier: the slabs a write off the devices is read from, and those for the part of a file a read does not want.
    size_t held() const {
        size_t n = staging_.held;
        for (const Reading& r : reads_) n += r.scratch.held;
        for (const Proof& p : proofs_) n += p.base.held + p.head.held + p.tail.held + p.state.held;
        return n;
    }

    // The file that makes a history of `n` tokens whole on disk, its state on a model that keeps one (`state`) and its last segment otherwise, or 0 where part of it is not there.
    uint64_t on_disk(const std::vector<std::string>& d, size_t block, size_t n, bool state) const {
        const DiskIndex::Path p = index_.path(d, block, n, state);
        if (!n || p.length != n) return 0;
        return state ? *p.state : p.pieces.back().key;
    }
    // The file holding a history's state at `n`, whatever of its blocks is on disk, or 0.
    uint64_t state_on_disk(const std::vector<std::string>& d, size_t block, size_t n) const {
        const auto s = index_.state_below(d, block, n);
        return s && s->first == n ? s->second : 0;
    }
    // A use of file `key` now.
    void renew(uint64_t key, bool back) {
        const auto now = DiskIndex::Time::clock::now();
        index_.touch(key, now, back);
        touch(key, now);
    }

    // What the calls finished since the last one left: the write in flight entered in the index, and each read that ended, whole or not.
    struct Settled {
        bool changed = false;   // the index gained or lost a file
        bool wrote = false;     // the write in flight ended, however
        std::vector<std::pair<uint64_t, bool>> reads;   // each read that ended, and whether every file of it read whole
    };
    Settled settle(infer::Model& model) {
        Settled out;
        for (const Finished& f : finished()) {
            if (f.read) {
                for (size_t i = 0; i < reads_.size(); ++i) {
                    Reading& r = reads_[i];
                    if (std::find(r.tickets.begin(), r.tickets.end(), f.ticket) == r.tickets.end()) continue;
                    // A read fails once, at its first file that does; those after it on the path fail with it.
                    if (!f.ok && r.ok) {
                        std::lock_guard<std::mutex> lk(m_);
                        ++errors_;
                    }
                    r.ok = r.ok && f.ok;
                    // A file that failed its read is gone, and what stood on it with it.
                    if (!f.ok && index_.remove(f.key)) {
                        out.changed = changed_ = true;
                        ++gone_[(size_t)Gone::read_failed];
                    }
                    if (--r.left) break;
                    out.reads.push_back({r.id, r.ok});
                    model.release_host(r.scratch);
                    // A proof's files are read: it goes on, or ends where a file failed, it was given up meanwhile or a build of its files failed another proof.
                    const auto proof = std::find_if(proofs_.begin(), proofs_.end(), [&](const Proof& p) { return p.id == r.id; });
                    if (proof != proofs_.end()) {
                        const bool doomed = std::any_of(proof->keys.begin(), proof->keys.end(), [&](uint64_t key) {
                            const DiskIndex::File* f = index_.find(key);
                            return !f || std::find(condemned_.begin(), condemned_.end(), f->unproved) != condemned_.end();
                        });
                        proof->read = true;
                        if (!r.ok || proof->dead || doomed) {
                            give_back(model, *proof);
                            proofs_.erase(proof);
                        }
                    }
                    reads_.erase(reads_.begin() + (std::ptrdiff_t)i);
                    break;
                }
                continue;
            }
            if (f.key != key_) continue;
            if (f.ok && cancelled_) {
                remove(f.key);
            } else if (f.ok) {
                pending_.key = f.key;
                changed_ = true;
                touch(f.key, pending_.used);
                written_ += pending_.bytes;
                // This build's own file makes the unproved ones it covers needless, but for one a proof is reading.
                const std::vector<uint64_t> held = pinned();
                for (uint64_t key : index_.covered(pending_))
                    if (std::find(held.begin(), held.end(), key) == held.end()) forget(key, Gone::covered);
                index_.add(std::move(pending_));
                out.changed = true;
                // Published: only now is the file a holder of its rows.
                ++whole_;
            } else {
                ++failed_;
            }
            pending_ = DiskIndex::File{};
            key_ = source_ = 0;
            cancelled_ = false;
            pins_.clear();
            model.release_host(staging_);
            out.wrote = true;
        }
        std::vector<uint64_t> unclaimed;
        {
            std::lock_guard<std::mutex> lk(m_);
            unclaimed.swap(unclaimed_);
        }
        for (uint64_t key : unclaimed) forget(key, Gone::unclaimed);
        if (sweep() || !unclaimed.empty()) out.changed = true;
        if (out.changed) prune(false);
        return out;
    }

    // Enters the files a server left under --disk-cache-keep in the index and drops those nothing reaches, those past the age limit and those over the cap, saying how many of each.
    // True once, when there were any (docs/DISK-TIER.md, Keeping entries across a restart).
    bool adopt() {
        const std::vector<DiskStore::Adopted> adopted = take_adopted();
        if (adopted.empty()) return false;
        size_t unread = 0, over = 0;
        for (const DiskStore::Adopted& a : adopted) {
            DiskIndex::File f;
            if (!DiskIndex::parse(a.blob, f)) {
                remove(a.key);
                ++unread;
                continue;
            }
            f.key = a.key;
            f.bytes = a.bytes;
            f.used = a.used;
            f.unproved = a.unproved ? a.from : 0;
            index_.add(std::move(f));
        }
        const size_t alone = prune(true);
        // The age limit by what the files hold: a file goes only when nothing beyond it was used within the limit (expire), so a conversation used since stands on files older than it.
        const size_t held = index_.files().size();
        expire();
        const size_t old = held - index_.files().size();
        while (index_.bytes() > cap()) {
            const std::optional<DiskIndex::Victim> v = index_.victim();
            if (!v) break;
            forget(v->key, Gone::cap);
            ++over;
        }
        std::fprintf(stderr, "server: %zu entries on disk from the server before", index_.files().size());
        if (const size_t n = unproved()) std::fprintf(stderr, ", %zu of them another build's, each proved at its first use", n);
        if (unread + alone + old + over) std::fprintf(stderr, "; dropped: %zu whose description did not read, %zu that no whole path reaches, %zu past the age limit, %zu over the cap", unread, alone, old, over);
        std::fprintf(stderr, "\n");
        return true;
    }

    // Files with nothing beyond them used for longer than the age limit deleted, a conversation going from its end down, but for those read or written on; true where one went.
    bool expire() {
        const uint64_t age = max_age();
        if (!age) return false;
        bool any = false;
        const DiskIndex::Time before = DiskIndex::Time::clock::now() - std::chrono::seconds(age);
        for (std::optional<DiskIndex::Victim> v; (v = index_.victim(before, pinned()));) {
            forget(v->key, Gone::age);
            any = true;
        }
        return any;
    }

    // The bytes a block of every device's KV storage and one state take in a file, measured once on slabs taken and given back.
    void measure(infer::Model& model) {
        if (measured_) return;
        measured_ = true;
        try {
            infer::HostHistory probe;
            model.alloc_host(model.kv_block_tokens(), probe, std::numeric_limits<size_t>::max(), true, 0, false);
            block_bytes_ = probe.bytes;
            model.release_host(probe);
            if (model.keeps_state()) {
                model.alloc_host(0, probe, std::numeric_limits<size_t>::max(), false);
                state_bytes_ = probe.bytes;
                model.release_host(probe);
            }
        } catch (const std::exception&) {}
    }
    // The bytes still to write of a history of `n` tokens: its segments not on disk, where `blocks` says its holder has them, and its state.
    uint64_t unwritten(const std::vector<std::string>& d, size_t block, size_t n, bool state, bool blocks) const {
        uint64_t bytes = 0;
        const auto missing = index_.missing(d, block, n);
        for (const auto& range : missing) bytes += blocks ? (range.second - range.first) / block * block_bytes_ : 0;
        if (state && !index_.has_state(d, block, n)) bytes += state_bytes_;
        return bytes;
    }

    enum class Wrote { nothing, started, refused };
    // Starts the write of the next file a history of `n` tokens lacks on disk, a segment and then its state, out of `held` or off the devices from `seq`, a segment of at most `most` rows from `seq`.
    // `source` and `bound` name the host entry whose slabs the store reads, 0 for slabs of its own (docs/src/server.md, the disk tier).
    Wrote write_next(infer::Model& model, const std::vector<std::string>& d, size_t n, bool back, DiskIndex::Time used, const infer::HostHistory* held, infer::Sequence* seq, uint64_t source, bool bound,
                     size_t most = std::numeric_limits<size_t>::max()) {
        const size_t block = model.kv_block_tokens();
        const bool state = model.keeps_state();
        const auto missing = index_.missing(d, block, n);
        DiskIndex::File f;
        f.back = back;
        f.used = used;
        std::vector<infer::HostRange> ranges;
        const infer::HostHistory* from = held;
        try {
            if (!missing.empty() && (!held || held->blocks)) {
                f.first = missing.front().first;
                f.end = missing.front().second;
                if (!held && most < f.end - f.first) f.end = f.first + std::max(block, most / block * block);
                f.below = d[f.first / block];
                f.ends.assign(d.begin() + (std::ptrdiff_t)(f.first / block) + 1, d.begin() + (std::ptrdiff_t)(f.end / block) + 1);
                if (!held) {
                    model.save_host_blocks(*seq, f.first, f.end, staging_, std::numeric_limits<size_t>::max());
                    from = &staging_;
                }
                ranges = model.host_ranges(*from, f.first, f.end);
            } else if (state && n && !index_.has_state(d, block, n)) {
                f.state = true;
                f.first = f.end = n;
                f.below = d[n / block];
                if (!held) {
                    model.save_host(*seq, n, staging_, std::numeric_limits<size_t>::max(), false);
                    from = &staging_;
                }
                ranges = model.host_state_ranges(*from);
            } else {
                return Wrote::nothing;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "server: %zu tokens of a history were not copied for the disk (%s)\n", n, e.what());
            model.release_host(staging_);
            return Wrote::refused;
        }
        const std::vector<size_t> layout = layout_of(ranges, from->slabs.size());
        std::string blob = DiskIndex::describe(f);
        f.bytes = DiskStore::file_bytes(blob.size(), layout);
        // What the file stands on stays while it is written.
        pins_.clear();
        for (const DiskIndex::Piece& p : index_.path(d, block, f.first, false).pieces) pins_.push_back(p.key);
        if (!room(f.bytes, back)) {
            pins_.clear();
            model.release_host(staging_);
            return Wrote::refused;
        }
        model.wait_host(*from);
        {
            std::lock_guard<std::mutex> lk(m_);
            in_flight_ = store_->put(std::move(blob), runs_of(*from, ranges), infer::Model::host_slab_bytes(), [this](bool ok, const std::string& error) {
                {
                    std::lock_guard<std::mutex> l(m_);
                    done_.push_back(Finished{in_flight_, false, ok, error, 0});
                }
                if (wake_) wake_();
            }, layout);
            key_ = in_flight_;
        }
        source_ = from == &staging_ ? 0 : source;
        bound_ = bound;
        cancelled_ = false;
        pending_ = std::move(f);
        return Wrote::started;
    }
    // The rows of the write in flight, a state's none.
    size_t pending_rows() const { return pending_.end - pending_.first; }
    // Whether the write in flight reads host entry `source`'s slabs and has not been stopped.
    bool writing_from(uint64_t source, bool bound) const { return key_ && source_ && source_ == source && bound_ == bound; }
    bool write_stopped() const { return cancelled_; }
    // The write in flight stopped: its file is not kept, even should it land before the cancel reaches it.
    void cancel_write() {
        if (!key_ || cancelled_) return;
        cancelled_ = true;
        cancel(key_);
    }

    // Starts the read of path `p` of a history into `target`, or with `state_only` of its state alone, and returns the read's id, which settle reports once every file has ended.
    // 0, nothing started, where host memory for the part not wanted is not to be had (docs/src/server.md, the disk tier).
    uint64_t read(infer::Model& model, const DiskIndex::Path& p, const infer::HostHistory& target, bool state_only) {
        Reading r;
        r.id = ++read_ids_;
        std::vector<Call> calls;
        try {
            calls_of(model, p, target, state_only, r.scratch, calls);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "server: a history of %zu tokens was not read from disk (%s)\n", p.length, e.what());
            model.release_host(r.scratch);
            return 0;
        }
        return send(std::move(r), calls);
    }

    // The read of path `p` streamed straight into `into`'s device storage (StreamedRead), its files pinned until end_stream; nothing where the model lays the path out otherwise than its files.
    std::unique_ptr<StreamedRead> stream(infer::Model& model, const DiskIndex::Path& p, infer::RestoreInto& into, uint64_t& id) {
        const infer::HostHistory image = model.host_layout(p.length);
        std::vector<StreamedRead::File> files;
        const auto segments = [&](StreamedRead::File& f, const std::vector<infer::HostRange>& ranges, const std::vector<infer::HostRange>& skip) {
            for (size_t i = 0; i < ranges.size(); ++i) {
                f.segments.push_back({ranges[i].device, ranges[i].offset, ranges[i].bytes, false});
                if (i < skip.size()) f.segments.push_back({skip[i].device, 0, skip[i].bytes, true});
            }
            f.layout.assign(image.device_bytes.size(), 0);
            for (const StreamedRead::File::Segment& s : f.segments) {
                f.layout[s.device] += s.bytes;
                f.payload += s.bytes;
            }
        };
        for (const DiskIndex::Piece& piece : p.pieces) {
            StreamedRead::File f;
            f.key = piece.key;
            std::vector<infer::HostRange> rest;
            // A file longer than the path's use of it: the rows past it, in its layout's order, are read and not sent.
            if (piece.use < piece.end) {
                const infer::HostHistory longer = model.host_layout(piece.end);
                for (const infer::HostRange& r : model.host_ranges(longer, piece.use, piece.end)) rest.push_back(r);
            }
            segments(f, model.host_ranges(image, piece.first, piece.use), rest);
            files.push_back(std::move(f));
        }
        if (p.state) {
            StreamedRead::File f;
            f.key = *p.state;
            segments(f, model.host_state_ranges(image), {});
            files.push_back(std::move(f));
        }
        std::vector<uint64_t> keys;
        for (const StreamedRead::File& f : files) keys.push_back(f.key);
        // The files are pinned only once the read exists, so a read that fails as it is made leaves no pin.
        std::unique_ptr<StreamedRead> read = std::make_unique<StreamedRead>(*store_, std::move(files), into, wake_);
        streams_.push_back({read_ids_ + 1, std::move(keys)});
        id = ++read_ids_;
        return read;
    }
    // A streamed read ended: its files unpinned, and the one whose read failed (`failed`, 0 for none) gone, as a read that fails takes its file.
    void end_stream(uint64_t id, uint64_t failed) {
        streams_.erase(std::remove_if(streams_.begin(), streams_.end(), [&](const Stream& s) { return s.id == id; }), streams_.end());
        if (!failed) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            ++errors_;
        }
        if (index_.remove(failed)) {
            changed_ = true;
            ++gone_[(size_t)Gone::read_failed];
        }
        store_->drop_entry(failed);
    }

    // ---- Another build's entries, adopted unproved and proved at their first use (docs/DISK-TIER.md, Adopting another build's entries). ----

    // What a history has in unproved files beyond the `held` tokens its request can have without them: its path through them to `end`, and the path to `fork`, where its request starts the rows it computes to compare.
    // `first` says its first block is another build's too and is compared before anything of the path is used, and `busy` names the proof already on one of its files.
    struct Plan {
        DiskIndex::Path path, base;
        size_t fork = 0, end = 0;
        bool first = false;
        uint64_t busy = 0;
        std::vector<uint64_t> keys, claim;   // its unproved files, and those of them the path uses whole, which an equal proof makes this build's
    };
    // The plan for the history whose digests are `d`, of `n` tokens, or none: nothing unproved beyond `held`, or a model that keeps a state (`state`) with no state on the path below its end to start from.
    std::optional<Plan> plan(const std::vector<std::string>& d, size_t block, size_t n, bool state, size_t held) const {
        if (!unproved()) return std::nullopt;
        Plan p;
        p.path = index_.path(d, block, n, state, true);
        p.end = p.path.length;
        if (p.end <= held || p.path.pieces.empty()) return std::nullopt;
        const auto note = [&](uint64_t key, bool whole) {
            const DiskIndex::File* f = index_.find(key);
            if (!f || !f->unproved) return;
            p.keys.push_back(key);
            if (whole) p.claim.push_back(key);
        };
        // The fork point: the highest position below the path's end at which a sequence of this model may be forked.
        p.base = state ? index_.path(d, block, p.end - 1, true, true) : index_.path(d, block, p.end - block, false, true);
        p.fork = p.base.length;
        if (state && !p.fork) return std::nullopt;
        for (const DiskIndex::Piece& piece : p.path.pieces) note(piece.key, piece.use == piece.end);
        if (p.path.state) note(*p.path.state, true);
        if (p.base.state) note(*p.base.state, true);
        if (p.keys.empty()) return std::nullopt;
        for (uint64_t key : p.keys)
            if (std::find(condemned_.begin(), condemned_.end(), index_.find(key)->unproved) != condemned_.end()) return std::nullopt;
        p.first = index_.find(p.path.pieces.front().key)->unproved != 0;
        if (p.fork <= held) p.base = DiskIndex::Path{};
        for (const Proof& proof : proofs_)
            for (uint64_t key : p.keys)
                if (std::find(proof.keys.begin(), proof.keys.end(), key) != proof.keys.end()) p.busy = proof.id;
        return p;
    }

    // Starts a plan's proof: the path to its fork read for its request to restore, and the files holding its first block, its last block and the state at its end read whole to compare with. The proof's id, which settle reports as a read, or 0.
    uint64_t prove(infer::Model& model, const Plan& plan) {
        Proof p;
        Reading r;
        r.id = p.id = ++read_ids_;
        p.keys = plan.keys;
        p.claim = plan.claim;
        p.fork = plan.fork;
        p.end = plan.end;
        p.first = plan.first;
        // The last comparison takes every block of the stretch the request computes that the file read whole for the path's end holds.
        p.from = std::max(plan.fork, plan.path.pieces.back().first);
        std::vector<Call> calls;
        const size_t most = std::numeric_limits<size_t>::max();
        const auto whole = [&](const DiskIndex::Piece& piece, infer::HostHistory& h) {
            model.alloc_host(piece.end, h, most, true, piece.first, false);
            const std::vector<infer::HostRange> ranges = model.host_ranges(h, piece.first, piece.end);
            calls.push_back(Call{piece.key, runs_of(h, ranges), layout_of(ranges, h.slabs.size())});
        };
        try {
            if (plan.base.length) {
                model.alloc_host(plan.base.length, p.base, most, true);
                calls_of(model, plan.base, p.base, false, r.scratch, calls);
            }
            const DiskIndex::Piece& root = plan.path.pieces.front();
            const DiskIndex::Piece& last = plan.path.pieces.back();
            if (plan.first) whole(root, p.head);
            if (!plan.first || last.key != root.key) whole(last, p.tail);
            if (plan.path.state) {
                model.alloc_host(plan.end, p.state, most, false);
                const std::vector<infer::HostRange> ranges = model.host_state_ranges(p.state);
                calls.push_back(Call{*plan.path.state, runs_of(p.state, ranges), layout_of(ranges, p.state.slabs.size())});
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "server: another build's history of %zu tokens was not read from disk to prove it (%s)\n", plan.end, e.what());
            model.release_host(r.scratch);
            give_back(model, p);
            return 0;
        }
        proofs_.push_back(std::move(p));
        return send(std::move(r), calls);
    }

    // Where proof `id` stands, or none once it has ended or been given up: whether its files are read, the block still to compare first (0 once it is), its fork and end, and the history to restore at the fork where the request holds less.
    struct Proving {
        bool read = false;
        size_t first = 0, fork = 0, end = 0;
        infer::HostHistory* base = nullptr;
        size_t from = 0;   // where the blocks of the last comparison begin
    };
    std::optional<Proving> proving(uint64_t id, size_t block) {
        for (Proof& p : proofs_)
            if (p.id == id && !p.dead) return Proving{p.read, p.first ? block : 0, p.fork, p.end, p.base.owner ? &p.base : nullptr, p.from};
        return std::nullopt;
    }

    // The next comparison of proof `id`, byte for byte against what the files hold: its request's first block, then its last with the state at the path's end; whether it was equal.
    // An equal last one makes the files this build's and gives the path the proof read to `proved`; an unequal one deletes them with their builds' other unproved files (docs/DISK-TIER.md, Adopting another build's entries).
    bool compare(uint64_t id, infer::Model& model, const infer::HostHistory& blocks, const infer::HostHistory& state, infer::HostHistory& proved) {
        const auto it = std::find_if(proofs_.begin(), proofs_.end(), [&](const Proof& p) { return p.id == id; });
        if (it == proofs_.end() || !it->read || it->dead) return false;
        Proof& p = *it;
        const size_t block = model.kv_block_tokens();
        model.wait_host(blocks);
        if (state.owner) model.wait_host(state);
        const infer::HostHistory& tail = p.tail.owner ? p.tail : p.head;
        bool equal;
        if (p.first) {
            equal = same(blocks, model.host_ranges(blocks, 0, block), p.head, model.host_ranges(p.head, 0, block));
            p.first = false;
            if (equal && p.end > block) return true;
        } else {
            equal = same(blocks, model.host_ranges(blocks, p.from, p.end), tail, model.host_ranges(tail, p.from, p.end));
            if (equal && p.state.owner) equal = state.owner && same(state, model.host_state_ranges(state), p.state, model.host_state_ranges(p.state));
        }
        if (equal) {
            for (uint64_t key : p.claim) {
                index_.prove(key);
                std::lock_guard<std::mutex> lk(m_);
                store_->claim(key, [this, key](bool ok, const std::string& error) {
                    if (!error.empty()) std::fprintf(stderr, "server: a proved entry was not made this build's on disk (%s)\n", error.c_str());
                    if (ok) return;
                    {
                        std::lock_guard<std::mutex> l(m_);
                        unclaimed_.push_back(key);
                    }
                    if (wake_) wake_();
                });
            }
            proved_ += p.claim.size();
            std::fprintf(stderr, "server: another build's history of %zu tokens proved this build's, %zu files\n", p.end, p.claim.size());
            proved = std::move(p.base);
            p.base = infer::HostHistory{};
        } else {
            for (uint64_t key : p.keys)
                if (const DiskIndex::File* f = index_.find(key)) condemned_.push_back(f->unproved);
            std::fprintf(stderr, "server: another build's history of %zu tokens is not what this build computes; its files and that build's other unproved files are deleted\n", p.end);
        }
        changed_ = true;
        give_back(model, p);
        proofs_.erase(it);
        sweep();
        return equal;
    }

    // Proof `id` given up, its request gone or computing without it; its files stay unproved.
    void abandon(uint64_t id, infer::Model& model) {
        const auto it = std::find_if(proofs_.begin(), proofs_.end(), [&](const Proof& p) { return p.id == id; });
        if (it == proofs_.end()) return;
        it->dead = true;
        if (!it->read) return;   // its slabs are still read into, and go when the read ends (settle)
        give_back(model, *it);
        proofs_.erase(it);
    }

    // The unproved files on disk.
    size_t unproved() const { return (size_t)std::count_if(index_.files().begin(), index_.files().end(), [](const DiskIndex::File& f) { return f.unproved != 0; }); }

    // At the stop: the slabs the tier's own calls hold given back, once the store is gone.
    void release(infer::Model& model) {
        model.release_host(staging_);
        for (Reading& r : reads_) model.release_host(r.scratch);
        reads_.clear();
        for (Proof& p : proofs_) give_back(model, p);
        proofs_.clear();
        key_ = source_ = 0;
    }

private:
    // A read in flight: the files it reads, the store's calls still out, and the slabs for the blocks of a file it wants only part of.
    struct Reading {
        uint64_t id = 0;
        std::vector<uint64_t> keys, tickets;
        size_t left = 0;
        bool ok = true;
        infer::HostHistory scratch;
    };

    // A proof in flight: the unproved files of its path and those an equal proof claims, its fork and end, whether its first block is still to compare, whether its read has ended, and whether it was given up before that.
    // `base` is the path to the fork, and `head`, `tail` and `state` the files holding the first block, the last block and the state at the end, read whole; one file holding both blocks is read once, as `head`.
    struct Proof {
        uint64_t id = 0;
        std::vector<uint64_t> keys, claim;
        size_t fork = 0, end = 0, from = 0;
        bool first = false, read = false, dead = false;
        infer::HostHistory base, head, tail, state;
    };
    void give_back(infer::Model& model, Proof& p) {
        for (infer::HostHistory* h : {&p.base, &p.head, &p.tail, &p.state}) model.release_host(*h);
    }
    // Whether two host histories hold the same bytes over ranges of the same sizes.
    static bool same(const infer::HostHistory& a, const std::vector<infer::HostRange>& ra, const infer::HostHistory& b, const std::vector<infer::HostRange>& rb) {
        if (ra.size() != rb.size()) return false;
        const size_t slab = infer::Model::host_slab_bytes();
        for (size_t i = 0; i < ra.size(); ++i) {
            if (ra[i].device != rb[i].device || ra[i].bytes != rb[i].bytes) return false;
            for (size_t at = 0; at < ra[i].bytes;) {
                const size_t oa = ra[i].offset + at, ob = rb[i].offset + at;
                const size_t n = std::min({ra[i].bytes - at, slab - oa % slab, slab - ob % slab});
                const auto* pa = (const uint8_t*)a.slabs[ra[i].device][oa / slab]->host_ptr() + oa % slab;
                const auto* pb = (const uint8_t*)b.slabs[rb[i].device][ob / slab]->host_ptr() + ob % slab;
                if (std::memcmp(pa, pb, n) != 0) return false;
                at += n;
            }
        }
        return true;
    }
    // The unproved files of a build one of whose entries failed its proof deleted, but for those a read is on, which go once it ends; whether any went.
    bool sweep() {
        if (condemned_.empty()) return false;
        const std::vector<uint64_t> held = pinned();
        std::vector<uint64_t> gone;
        for (const DiskIndex::File& f : index_.files())
            if (f.unproved && std::find(condemned_.begin(), condemned_.end(), f.unproved) != condemned_.end() && std::find(held.begin(), held.end(), f.key) == held.end())
                gone.push_back(f.key);
        for (uint64_t key : gone) forget(key, Gone::condemned);
        disproved_ += gone.size();
        if (!gone.empty()) prune(false);
        return !gone.empty();
    }

    // One file's read: its key, where its bytes go and the layout it must have.
    struct Call {
        uint64_t key;
        std::vector<StoreRun> runs;
        std::vector<size_t> layout;
    };
    // The reads of path `p` into `target`, a file the path uses part of taking `scratch` for the rest, appended to `calls`.
    void calls_of(infer::Model& model, const DiskIndex::Path& p, const infer::HostHistory& target, bool state_only, infer::HostHistory& scratch, std::vector<Call>& calls) {
        for (const DiskIndex::Piece& piece : state_only ? std::vector<DiskIndex::Piece>{} : p.pieces) {
            std::vector<infer::HostRange> ranges = model.host_ranges(target, piece.first, piece.use);
            std::vector<StoreRun> runs = runs_of(target, ranges);
            if (piece.use < piece.end) {
                model.alloc_host(piece.end, scratch, std::numeric_limits<size_t>::max(), true, piece.use, false);
                const std::vector<infer::HostRange> rest = model.host_ranges(scratch, piece.use, piece.end);
                const std::vector<StoreRun> more = runs_of(scratch, rest);
                std::vector<StoreRun> both;
                for (size_t i = 0; i < runs.size(); ++i) {
                    both.push_back(runs[i]);
                    both.push_back(more[i]);
                }
                runs.swap(both);
                ranges.insert(ranges.end(), rest.begin(), rest.end());
            }
            calls.push_back(Call{piece.key, std::move(runs), layout_of(ranges, target.slabs.size())});
        }
        if (p.state) {
            const std::vector<infer::HostRange> ranges = model.host_state_ranges(target);
            calls.push_back(Call{*p.state, runs_of(target, ranges), layout_of(ranges, target.slabs.size())});
        }
    }
    // Read `r`'s calls handed to the store; its id.
    uint64_t send(Reading r, std::vector<Call>& calls) {
        const uint64_t id = r.id;
        r.left = calls.size();
        std::lock_guard<std::mutex> lk(m_);
        for (Call& c : calls) {
            const uint64_t ticket = ++tickets_, key = c.key;
            r.tickets.push_back(ticket);
            r.keys.push_back(key);
            store_->get(key, std::move(c.runs), infer::Model::host_slab_bytes(), [this, key, ticket](bool ok, const std::string& error) {
                {
                    std::lock_guard<std::mutex> l(m_);
                    done_.push_back(Finished{key, true, ok, error, ticket});
                }
                if (wake_) wake_();
            }, std::move(c.layout));
        }
        reads_.push_back(std::move(r));
        return id;
    }

    // The runs of ranges of a host history, as the store takes them, and the layout a file of them records: the bytes a device.
    static std::vector<StoreRun> runs_of(const infer::HostHistory& h, const std::vector<infer::HostRange>& ranges) {
        std::vector<StoreRun> runs;
        for (const infer::HostRange& r : ranges) {
            StoreRun run;
            run.slabs = h.slabs[r.device];
            run.offset = r.offset;
            run.bytes = r.bytes;
            runs.push_back(std::move(run));
        }
        return runs;
    }
    static std::vector<size_t> layout_of(const std::vector<infer::HostRange>& ranges, size_t devices) {
        std::vector<size_t> layout(devices, 0);
        for (const infer::HostRange& r : ranges) layout[r.device] += r.bytes;
        return layout;
    }

    // The files that may not go now: those a read is on, and those the file being written stands on.
    std::vector<uint64_t> pinned() const {
        std::vector<uint64_t> keys = pins_;
        for (const Reading& r : reads_) keys.insert(keys.end(), r.keys.begin(), r.keys.end());
        for (const Stream& s : streams_) keys.insert(keys.end(), s.keys.begin(), s.keys.end());
        return keys;
    }
    void forget(uint64_t key, Gone why) {
        ++gone_[(size_t)why];
        changed_ = true;
        index_.remove(key);
        remove(key);
    }
    // Files no whole path reaches any more deleted, states whose history's blocks are not all there only at a start (`states`), when no history in memory can still use them; how many.
    size_t prune(bool states) {
        const std::vector<uint64_t> gone = index_.unreachable(states);
        for (uint64_t key : gone) forget(key, Gone::unreachable);
        return gone.size();
    }
    // Disk room for `bytes` more within the cap, files going as the index orders them (DiskIndex::victim); one of a conversation that did not come back (`back` false) takes no room of one that did.
    // False, nothing removed, where the room cannot be made.
    bool room(uint64_t bytes, bool back) {
        if (bytes > cap()) return false;
        if (index_.bytes() + bytes <= cap()) return true;
        DiskIndex trial = index_;
        std::vector<uint64_t> gone;
        const std::vector<uint64_t> keep = pinned();
        while (trial.bytes() + bytes > cap()) {
            const std::optional<DiskIndex::Victim> v = trial.victim(std::nullopt, keep);
            if (!v || (!back && v->back)) return false;
            trial.remove(v->key);
            gone.push_back(v->key);
        }
        for (uint64_t key : gone) forget(key, Gone::cap);
        capped_ += gone.size();
        return true;
    }

    DiskOptions options_;
    std::function<void()> wake_;
    std::thread starter_;
    mutable std::mutex m_;
    std::unique_ptr<DiskStore> store_;   // under m_, once made
    bool ready_ = false, writing_ = false, adopted_taken_ = false;   // under m_
    std::atomic<size_t> refused_dirs_{0};   // set once as the store is made
    uint64_t in_flight_ = 0;             // under m_, the key of the write in flight
    size_t errors_ = 0;                  // under m_
    std::chrono::steady_clock::time_point stopped_;   // under m_, when writing stopped or was last checked
    std::vector<Finished> done_;         // under m_
    uint64_t tickets_ = 0;               // under m_
    // The scheduler's thread's, under its lock: the index, the file being written with the host entry or the slabs it is read from and the files it stands on, the reads in flight, and what was written and deleted for the cap.
    DiskIndex index_;
    bool changed_ = false;
    DiskIndex::File pending_;
    uint64_t key_ = 0, source_ = 0;
    bool bound_ = false, cancelled_ = false;
    infer::HostHistory staging_;
    std::vector<uint64_t> pins_;
    std::vector<Reading> reads_;
    // A streamed read's files, pinned until it ends (end_stream).
    struct Stream {
        uint64_t id;
        std::vector<uint64_t> keys;
    };
    std::vector<Stream> streams_;
    std::vector<Proof> proofs_;
    std::vector<uint64_t> condemned_;    // the builds one of whose entries failed its proof
    std::vector<uint64_t> unclaimed_;    // under m_, the proved files the store could not make this build's
    size_t proved_ = 0, disproved_ = 0;
    uint64_t read_ids_ = 0, written_ = 0, said_ = 0;
    size_t capped_ = 0;
    std::array<size_t, (size_t)Gone::count> gone_{};   // Figures::gone
    uint64_t whole_ = 0, failed_ = 0;                 // Figures::whole and failed
    bool measured_ = false;
    uint64_t block_bytes_ = 0, state_bytes_ = 0;
};

} // namespace server
