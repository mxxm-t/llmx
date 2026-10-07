#pragma once
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
    return "model: " + digest + "\nnumerics: " + build.numerics + "\ncompiler: " + build.compiler + "\nflags: " + build.flags + "\nshaders: " + build.shaders + "\nlibm: " + build.libm + "\nlayout: " + flat + "\n";
}

// The server's disk tier, from --disk-cache-bytes, --disk-cache-dir, --disk-cache-floor, --disk-cache-keep and --disk-cache-max-age; `bytes` 0 keeps none.
struct DiskOptions {
    uint64_t bytes = 0;
    std::string dir;
    std::optional<uint64_t> floor;   // none takes the larger of 16 GiB and a twentieth of the file system (check_disk_cache)
    bool keep = false;               // at a clean exit flush memory to disk and leave the entries for the next server, which adopts them
    uint64_t max_age = 24 * 3600;    // seconds an entry may go unused before it is deleted, 0 for no limit
    std::string model_path;          // the model file whose digest every entry's identity carries
    std::chrono::milliseconds pace{0};   // DiskStore::Options::pace, for tests
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
// How long a server under --disk-cache-keep has had nothing to do before it writes ahead what a stop would have to, so a turn that follows at once meets no copy off the devices.
constexpr std::chrono::seconds kDiskIdle{5};

class DiskTier {
public:
    // A call the store has finished: its key, whether it was a read, whether it succeeded, and why not.
    struct Finished {
        uint64_t key = 0;
        bool read = false, ok = false;
        std::string error;
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
    // Stops the write in flight; the store then removes its directory.
    ~DiskTier() {
        if (starter_.joinable()) starter_.join();
        std::unique_ptr<DiskStore> store;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (store_ && in_flight_) store_->cancel(in_flight_);
            store = std::move(store_);
        }
        store.reset();
    }
    DiskTier(const DiskTier&) = delete;
    DiskTier& operator=(const DiskTier&) = delete;

    uint64_t cap() const { return options_.bytes; }
    bool keeps() const { return options_.keep; }
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

    // The bytes an entry of a blob of `blob_bytes` over host history `h` takes on disk.
    static uint64_t file_bytes(size_t blob_bytes, const infer::HostHistory& h) { return DiskStore::file_bytes(blob_bytes, h.device_bytes); }

    // Starts the write of host history `h` with `blob` and returns its key, after can_write; `h`'s slabs must not change until finished reports the key.
    uint64_t write(std::string blob, const infer::HostHistory& h, size_t slab) {
        std::vector<StoreRun> runs = runs_of(h);
        std::lock_guard<std::mutex> lk(m_);
        // The key is not known until put returns, so the callback names the write in flight, which no other write replaces before finished has reported it.
        in_flight_ = store_->put(std::move(blob), std::move(runs), slab, [this](bool ok, const std::string& error) {
            {
                std::lock_guard<std::mutex> l(m_);
                done_.push_back(Finished{in_flight_, false, ok, error});
            }
            if (wake_) wake_();
        });
        return in_flight_;
    }

    // Starts the read of entry `key` into host history `h`, whose layout must be the entry's, after the store is made; finished reports it, a read that fails having deleted the entry.
    void read(uint64_t key, const infer::HostHistory& h, size_t slab) {
        std::lock_guard<std::mutex> lk(m_);
        store_->get(key, runs_of(h), slab, [this, key](bool ok, const std::string& error) {
            {
                std::lock_guard<std::mutex> l(m_);
                done_.push_back(Finished{key, true, ok, error});
            }
            if (wake_) wake_();
        });
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
            if (f.read) {
                if (!f.ok) ++errors_;
                continue;
            }
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

private:
    static std::vector<StoreRun> runs_of(const infer::HostHistory& h) {
        std::vector<StoreRun> runs(h.slabs.size());
        for (size_t i = 0; i < h.slabs.size(); ++i) {
            runs[i].slabs = h.slabs[i];
            runs[i].bytes = h.device_bytes[i];
        }
        return runs;
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
};

} // namespace server
