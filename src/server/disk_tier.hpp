#pragma once
#include <algorithm>
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

// The server's disk tier, from --disk-cache-bytes, --disk-cache-dir, --disk-cache-floor, --disk-cache-keep and --disk-cache-max-age; `bytes` 0 keeps none.
// How long a server under --disk-cache-keep has had nothing to do before it writes ahead what a stop would have to, so a turn that follows at once meets no copy off the devices.
constexpr std::chrono::seconds kDiskIdle{5};

struct DiskOptions {
    uint64_t bytes = 0;
    std::string dir;
    std::optional<uint64_t> floor;   // none takes the larger of 16 GiB and a twentieth of the file system (check_disk_cache)
    bool keep = false;               // at a clean exit flush memory to disk and leave the entries for the next server, which adopts them
    uint64_t max_age = 24 * 3600;    // seconds an entry may go unused before it is deleted, 0 for no limit
    std::string model_path;          // the model file whose digest every entry's identity carries
    std::chrono::milliseconds pace{0};   // DiskStore::Options::pace, for tests
    std::chrono::milliseconds idle{kDiskIdle};   // how long nothing runs before the idle writes start, shorter in tests
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

class DiskTier {
public:
    // A call the store has finished: its key, whether it was a read, whether it succeeded, and why not.
    struct Finished {
        uint64_t key = 0;
        bool read = false, ok = false;
        std::string error;
        uint64_t ticket = 0;   // a read's own number, since two reads may be on one file
    };

    // Reads the model file's digest on a thread of its own, the cached one where its stamp is unchanged, then makes the store under options.dir with the identity of the digest, the build and the model's host layout (Model::host_identity); nothing is written before that.
    // `wake` runs, under no lock of the tier's, when the store is made or refused and when a write finishes.
    DiskTier(const DiskOptions& options, const infer::Model& model, std::function<void()> wake) : options_(options), wake_(std::move(wake)) {
        const std::string layout = model.host_identity();
        starter_ = std::thread([this, layout] {
            std::unique_ptr<DiskStore> store;
            try {
                const std::string digest = format::cached_file_sha256((std::filesystem::u8path(options_.dir) / "digests").u8string(), options_.model_path);
                const std::string text = disk_identity(digest, layout, build_facts());
                core::Sha id(true);
                id.update(text.data(), text.size());
                const std::string hex = id.hex();
                std::array<uint8_t, 32> identity{};
                for (size_t i = 0; i < identity.size(); ++i) identity[i] = (uint8_t)std::stoul(hex.substr(2 * i, 2), nullptr, 16);
                DiskStore::Options o;
                o.root = options_.dir;
                o.floor = options_.floor.value_or(0);
                o.keep = options_.keep;
                o.max_age = options_.max_age;
                o.pace = options_.pace;
                o.identity = text;
                store = std::make_unique<DiskStore>(o, identity);
                std::fprintf(stderr, "server: disk cache in %s, writes %s the file cache, numerics %.16s, %zu entries adopted\n", store->directory().u8string().c_str(),
                             store->direct() ? "around" : "through", LLMX_NUMERICS, store->adopted().size());
                for (const std::string& why : store->refused()) std::fprintf(stderr, "server: a kept directory was not adopted, %s\n", why.c_str());
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
    struct Figures {
        size_t files = 0, states = 0, capped = 0;
        uint64_t bytes = 0, written = 0;
    };
    Figures figures() const { return {index_.files().size(), index_.states(), capped_, index_.bytes(), written_}; }
    // Once an idle server's writes have ended: one line of what went to disk since the last such line.
    void say_written() {
        if (written_ == said_) return;
        std::fprintf(stderr, "server: %.1f MiB written to disk since the idle writes last ended; %zu files on disk, %zu of them states\n", (double)(written_ - said_) / (1 << 20), index_.files().size(), index_.states());
        said_ = written_;
    }
    size_t calls() const { return (key_ ? 1 : 0) + reads_.size(); }
    // The host memory the tier's own calls hold beyond the host tier: the slabs a write off the devices is read from, and those for the part of a file a read does not want.
    size_t held() const {
        size_t n = staging_.held;
        for (const Reading& r : reads_) n += r.scratch.held;
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
                    if (!f.ok && index_.remove(f.key)) out.changed = changed_ = true;
                    if (--r.left) break;
                    out.reads.push_back({r.id, r.ok});
                    model.release_host(r.scratch);
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
                index_.add(std::move(pending_));
                out.changed = true;
            }
            pending_ = DiskIndex::File{};
            key_ = source_ = 0;
            cancelled_ = false;
            pins_.clear();
            model.release_host(staging_);
            out.wrote = true;
        }
        if (out.changed) prune(false);
        return out;
    }

    // The files a server left under --disk-cache-keep, once the store has adopted them: into the index by their descriptions, those nothing reaches and those over the cap dropped, and a line saying how many each rule dropped; true once, when there were any.
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
            index_.add(std::move(f));
        }
        const size_t alone = prune(true);
        while (index_.bytes() > cap()) {
            const std::optional<DiskIndex::Victim> v = index_.victim();
            if (!v) break;
            forget(v->key);
            ++over;
        }
        std::fprintf(stderr, "server: %zu entries on disk from the server before", index_.files().size());
        if (unread + alone + over) std::fprintf(stderr, "; dropped: %zu whose description did not read, %zu that no whole path reaches, %zu over the cap", unread, alone, over);
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
            forget(v->key);
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
    // Starts the write of the next file a history of `n` tokens lacks on disk, after can_write: its first missing segment, out of `held`, a copy in host memory holding its blocks, or copied off the devices from `seq`; then, on a model that keeps a state, its state, out of `held` or `seq`'s checkpoint.
    // A holder of a state alone writes the state, whatever of the history's blocks is on disk: a history in memory forks it meanwhile, and whoever holds the blocks writes them.
    // `source` and `bound` name the host entry whose slabs the store reads, 0 where it reads slabs of its own, taken beyond the host tier for the write's time; refused where disk room or host memory is not to be had.
    Wrote write_next(infer::Model& model, const std::vector<std::string>& d, size_t n, bool back, DiskIndex::Time used, const infer::HostHistory* held, infer::Sequence* seq, uint64_t source, bool bound) {
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
    // Whether the write in flight reads host entry `source`'s slabs and has not been stopped.
    bool writing_from(uint64_t source, bool bound) const { return key_ && source_ && source_ == source && bound_ == bound; }
    bool write_stopped() const { return cancelled_; }
    // The write in flight stopped: its file is not kept, even should it land before the cancel reaches it.
    void cancel_write() {
        if (!key_ || cancelled_) return;
        cancelled_ = true;
        cancel(key_);
    }

    // Starts the read of path `p` of a history into `target`, slabs for its length: every segment into its blocks' places, one the history parts from inside with its further blocks into slabs of the read's own, and the state into its place; with `state_only`, the state alone into a target that holds no blocks.
    // Returns the read's id, which settle reports once every file has ended; 0, nothing started, where host memory for the part not wanted is not to be had.
    uint64_t read(infer::Model& model, const DiskIndex::Path& p, const infer::HostHistory& target, bool state_only) {
        Reading r;
        r.id = ++read_ids_;
        struct Call {
            uint64_t key;
            std::vector<StoreRun> runs;
            std::vector<size_t> layout;
        };
        std::vector<Call> calls;
        try {
            for (const DiskIndex::Piece& piece : state_only ? std::vector<DiskIndex::Piece>{} : p.pieces) {
                std::vector<infer::HostRange> ranges = model.host_ranges(target, piece.first, piece.use);
                std::vector<StoreRun> runs = runs_of(target, ranges);
                if (piece.use < piece.end) {
                    model.alloc_host(piece.end, r.scratch, std::numeric_limits<size_t>::max(), true, piece.use, false);
                    const std::vector<infer::HostRange> rest = model.host_ranges(r.scratch, piece.use, piece.end);
                    const std::vector<StoreRun> more = runs_of(r.scratch, rest);
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
        } catch (const std::exception& e) {
            std::fprintf(stderr, "server: a history of %zu tokens was not read from disk (%s)\n", p.length, e.what());
            model.release_host(r.scratch);
            return 0;
        }
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

    // At the stop: the slabs the tier's own calls hold given back, once the store is gone.
    void release(infer::Model& model) {
        model.release_host(staging_);
        for (Reading& r : reads_) model.release_host(r.scratch);
        reads_.clear();
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
        return keys;
    }
    void forget(uint64_t key) {
        changed_ = true;
        index_.remove(key);
        remove(key);
    }
    // Files no whole path reaches any more deleted, states whose history's blocks are not all there only at a start (`states`), when no history in memory can still use them; how many.
    size_t prune(bool states) {
        const std::vector<uint64_t> gone = index_.unreachable(states);
        for (uint64_t key : gone) forget(key);
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
        for (uint64_t key : gone) forget(key);
        capped_ += gone.size();
        return true;
    }

    DiskOptions options_;
    std::function<void()> wake_;
    std::thread starter_;
    mutable std::mutex m_;
    std::unique_ptr<DiskStore> store_;   // under m_, once made
    bool ready_ = false, writing_ = false, adopted_taken_ = false;   // under m_
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
    uint64_t read_ids_ = 0, written_ = 0, said_ = 0;
    size_t capped_ = 0;
    bool measured_ = false;
    uint64_t block_bytes_ = 0, state_bytes_ = 0;
};

} // namespace server
