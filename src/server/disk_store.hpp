#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "backends/backend.hpp"
#include "core/crc32c.hpp"
#include "core/host_memory.hpp"
#include "core/sha.hpp"
#include "format/file_reader.hpp"
#include "format/file_writer.hpp"
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

// The disk tier's store (docs/DISK-TIER.md): entry files in a directory of the server's own under a root, held by a lock while the server runs, written and read on a thread of its own.
// It only moves bytes: what is kept, renewed, demoted or dropped is the scheduler's.

namespace server {

// A run of an entry, as host memory holds it: `bytes` at `offset` of consecutive host-visible slabs of the store call's slab size taken end to end (infer::HostHistory, infer::HostRange).
// An entry's payload is its runs one after another, so the same payload can come from, or go to, memory laid out in other runs.
struct StoreRun {
    std::vector<backend::BufferPtr> slabs;
    size_t bytes = 0;
    size_t offset = 0;
};

class DiskStore {
public:
    struct Options {
        std::string root;            // the directory every server's own directory sits under (--disk-cache-dir)
        uint64_t floor = 0;          // free space the file system keeps after every write (--disk-cache-floor)
        bool keep = false;           // at a clean exit leave the entries for the next server, and at start adopt those other servers left (--disk-cache-keep)
        std::string identity;        // the text the identity is the digest of, a `name: value` line a component, left in the directory so a server that adopts nothing can say which differs; none leaves no file
        bool proved = false;         // adopt, unproved, the entries of a build other than this one whose model, layout and entry version are this store's (--disk-cache-adopt proved)
        uint64_t max_age = 0;        // seconds after an entry's last use when sweeps and adoption delete it, 0 for never (--disk-cache-max-age)
        std::chrono::milliseconds pace{0};   // a pause after each chunk written or read, a write's cut short by a cancel, which tests hold a call in flight with
        std::function<bool(const char* step)> fault;   // a step of a claim or an adoption fails where this says so (claim-sync, provenance, repair), for tests
    };
    // What a finished call reports: whether it succeeded, and if not, why.
    using Done = std::function<void(bool ok, const std::string& error)>;
    // An entry adopted at start from a directory another server left: its key in this store, its blob, its runs' bytes, its file's bytes and its last use.
    struct Adopted {
        uint64_t key = 0;
        std::string blob;
        std::vector<size_t> run_bytes;
        uint64_t bytes = 0;
        std::filesystem::file_time_type used;
        bool unproved = false;   // written by another build: read as its bytes, and this store's only once claimed
        uint64_t from = 0;       // which build, one number an identity, 0 for this one
    };

    // The entries' alignment on disk, which every direct-I/O granule up to it divides, and the unit a checksum covers.
    static constexpr size_t kAlign = size_t(1) << 20, kChunk = size_t(4) << 20;
    // The version of a file's layout and of what its description holds; a file of another is not read.
    // 2: a file is a segment or a state of a history (DiskIndex::describe), where it was a whole copy.
    static constexpr uint32_t kVersion = 2;

    // Makes the server's own directory under the root, locked for the store's life, after sweeping the root and, with keep, adopting what servers of this identity left.
    // It then chooses the write mode by a short probe and starts the I/O thread.
    DiskStore(Options options, const std::array<uint8_t, 32>& identity) : options_(std::move(options)), identity_(identity), staging_(kChunk) {
        const auto root = std::filesystem::u8path(options_.root);
        std::filesystem::create_directories(root);
        std::random_device random;
        for (int attempt = 0; attempt < 32 && dir_.empty(); ++attempt) {
            const auto candidate = root / ("server-" + std::to_string(process_id()) + "-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) dir_ = candidate;
        }
        if (dir_.empty()) throw std::runtime_error("disk cache: cannot make a directory under " + options_.root);
        owner_only(dir_);
        lock_ = lock(dir_);
        if (!lock_.held()) throw std::runtime_error("disk cache: cannot lock " + dir_.u8string());
        if (!options_.identity.empty()) std::ofstream(dir_ / "identity", std::ios::binary) << options_.identity;
        sweep(true);
        probe();
        thread_ = std::thread([this] { run(); });
    }

    // Stops the I/O thread, cancelling what is queued, then leaves the directory, unlocked and marked kept, for the next server with keep, or removes it.
    ~DiskStore() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stopping_ = true;
            for (auto& j : jobs_) j->cancelled = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        std::error_code ec;
        if (options_.keep) {
            std::ofstream(dir_ / "kept").put('\n');
            lock_ = Lock{};
        } else {
            lock_ = Lock{};
            std::filesystem::remove_all(dir_, ec);
        }
    }
    DiskStore(const DiskStore&) = delete;
    DiskStore& operator=(const DiskStore&) = delete;

    const std::filesystem::path& directory() const { return dir_; }
    // Whether entries are written around the file cache, as the probe chose.
    bool direct() const { return direct_; }
    const std::vector<Adopted>& adopted() const { return adopted_; }
    // For each kept directory of another identity met at start, why it gave nothing: the components that differ, or that it was written before identities were kept.
    const std::vector<std::string>& refused() const { return refused_; }

    // Queues an entry's write and returns its key; `done` runs on the I/O thread once the file is in place, or with the reason it is not, and the slabs must not change until then.
    // `layout` is what the file records of its payload, a size a device; the runs are one stream and need not end where its sizes do (docs/src/server.md, the disk tier).
    uint64_t put(std::string blob, std::vector<StoreRun> runs, size_t slab, Done done, std::vector<size_t> layout = {}) {
        auto j = std::make_shared<Job>();
        j->put = true;
        j->blob = std::move(blob);
        j->layout = layout_of(runs, std::move(layout));
        j->runs = std::move(runs);
        j->slab = slab;
        j->done = std::move(done);
        std::lock_guard<std::mutex> lk(m_);
        j->key = ++keys_;
        jobs_.push_back(j);
        cv_.notify_all();
        return j->key;
    }

    // Queues a read of entry `key` into the runs' slabs, which must add up to the file's `layout`; reads go before writes.
    // `done` reports a missing entry, another identity or layout, or a failed checksum or read, after which the entry is deleted.
    void get(uint64_t key, std::vector<StoreRun> runs, size_t slab, Done done, std::vector<size_t> layout = {}) {
        auto j = std::make_shared<Job>();
        j->key = key;
        j->layout = layout_of(runs, std::move(layout));
        j->runs = std::move(runs);
        j->slab = slab;
        j->done = std::move(done);
        std::lock_guard<std::mutex> lk(m_);
        jobs_.push_back(j);
        cv_.notify_all();
    }

    // The 32 bytes an entry's header carries for the identity text `text`: its SHA-256.
    static std::array<uint8_t, 32> identity_of(const std::string& text) {
        core::Sha id(true);
        id.update(text.data(), text.size());
        const std::string hex = id.hex();
        std::array<uint8_t, 32> bytes{};
        for (size_t k = 0; k < bytes.size(); ++k) bytes[k] = (uint8_t)std::stoul(hex.substr(2 * k, 2), nullptr, 16);
        return bytes;
    }

    // Makes an unproved entry this store's own, its header rewritten under this identity, once its bytes are proved to be what this build computes (docs/DISK-TIER.md, Adopting another build's entries).
    // Queued like a read; a claim that did not reach the disk keeps the entry and says so in `done`'s error, and only a header left whole under neither build fails it, the entry deleted.
    void claim(uint64_t key, Done done) {
        auto j = std::make_shared<Job>();
        j->key = key;
        j->claim = true;
        j->done = std::move(done);
        std::lock_guard<std::mutex> lk(m_);
        jobs_.push_back(j);
        cv_.notify_all();
    }

    // Stops a queued or running write of `key`, whose `done` then reports the cancel; false when there is none.
    bool cancel(uint64_t key) {
        std::lock_guard<std::mutex> lk(m_);
        for (auto& j : jobs_)
            if (j->put && j->key == key) {
                j->cancelled = true;
                return true;
            }
        if (running_ && running_->put && running_->key == key) {
            running_->cancelled = true;
            cv_.notify_all();
            return true;
        }
        return false;
    }

    // Removes entry `key`, cancelling its write if one is queued or running.
    void evict(uint64_t key) {
        cancel(key);
        std::lock_guard<std::mutex> lk(m_);
        if (!sizes_.erase(key)) return;
        std::error_code ec;
        std::filesystem::remove(path(key, ".kv"), ec);
        std::filesystem::remove(path(key, ".from"), ec);
        foreign_.erase(key);
    }

    // Records entry `key`'s last use, which a restart's adoption and the age limit read.
    void touch(uint64_t key, std::filesystem::file_time_type used) {
        std::error_code ec;
        std::filesystem::last_write_time(path(key, ".kv"), used, ec);
    }

    // The bytes a second the store has written of late, flushed: the start's probe, then each entry written, half and half; 0 where the probe failed.
    double write_rate() const { return rate_.load(); }

    // What the file system has free for this process.
    std::optional<uint64_t> free_bytes() const {
        std::error_code ec;
        const auto s = std::filesystem::space(dir_, ec);
        if (ec) return std::nullopt;
        return (uint64_t)s.available;
    }

    // The bytes an entry of `blob_bytes` and these runs takes on disk.
    static uint64_t file_bytes(size_t blob_bytes, const std::vector<size_t>& run_bytes) {
        uint64_t payload = 0;
        for (size_t b : run_bytes) payload += b;
        return header_bytes(blob_bytes, run_bytes.size(), chunks(payload)) + round_up(payload, kAlign);
    }

    // Removes every directory under the root whose lock is free, a server's that ended, but a kept one younger than the age limit; with `adopt` it first adopts those of this store's identity.
    // A directory whose lock is held is a live server's and is never touched.
    void sweep(bool adopt = false) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(std::filesystem::u8path(options_.root), ec)) {
            const auto other = e.path();
            if (other == dir_ || !e.is_directory(ec) || other.filename().u8string().rfind("server-", 0) != 0) continue;
            Lock held = lock(other);
            if (!held.held()) continue;
            const bool kept = std::filesystem::exists(other / "kept", ec);
            // A directory an adoption could not finish with keeps what a later start can still recover, until its age limit.
            if (adopt && options_.keep) {
                if (adopt_from(other) && kept && !expired(other / "kept")) continue;
            } else if (kept && !expired(other / "kept")) continue;
            held = Lock{};
            std::filesystem::remove_all(other, ec);
        }
    }

private:
    // A call's layout: the one given, which its runs must add up to, or the runs' own sizes.
    static std::vector<size_t> layout_of(const std::vector<StoreRun>& runs, std::vector<size_t> layout) {
        size_t held = 0, said = 0;
        for (const StoreRun& r : runs) held += r.bytes;
        if (layout.empty())
            for (const StoreRun& r : runs) layout.push_back(r.bytes);
        for (size_t b : layout) said += b;
        if (held != said) throw std::logic_error("disk cache: runs that do not add up to the entry's layout");
        return layout;
    }

    struct Job {
        bool put = false, claim = false;
        uint64_t key = 0;
        std::string blob;
        std::vector<size_t> layout;
        std::vector<StoreRun> runs;
        size_t slab = 0;
        Done done;
        std::atomic<bool> cancelled{false};
    };

    // A directory's lock: a file in it held locked, released by the kernel however the process ends.
    class Lock {
    public:
        Lock() = default;
        Lock(Lock&& o) noexcept { swap(o); }
        Lock& operator=(Lock&& o) noexcept {
            Lock t(std::move(o));
            swap(t);
            return *this;
        }
        ~Lock() { release(); }
        bool held() const {
#if defined(_WIN32)
            return file_ != INVALID_HANDLE_VALUE;
#else
            return fd_ >= 0;
#endif
        }
#if defined(_WIN32)
        explicit Lock(HANDLE f) : file_(f) {}
#else
        explicit Lock(int fd) : fd_(fd) {}
#endif
    private:
        void swap(Lock& o) noexcept {
#if defined(_WIN32)
            std::swap(file_, o.file_);
#else
            std::swap(fd_, o.fd_);
#endif
        }
        void release() noexcept {
#if defined(_WIN32)
            if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
#else
            if (fd_ >= 0) ::close(fd_);
            fd_ = -1;
#endif
        }
#if defined(_WIN32)
        HANDLE file_ = INVALID_HANDLE_VALUE;
#else
        int fd_ = -1;
#endif
    };

    // The lock of directory `d`, or one not held when another process holds it.
    static Lock lock(const std::filesystem::path& d) {
        const auto file = d / "lock";
#if defined(_WIN32)
        HANDLE h = CreateFileW(file.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return Lock{};
        OVERLAPPED ov{};
        if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) {
            CloseHandle(h);
            return Lock{};
        }
        return Lock(h);
#else
        const int fd = ::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) return Lock{};
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd);
            return Lock{};
        }
        return Lock(fd);
#endif
    }

    static unsigned long process_id() {
#if defined(_WIN32)
        return GetCurrentProcessId();
#else
        return (unsigned long)getpid();
#endif
    }

    // Readable and writable by its owner alone; on Windows the directory under the root inherits the root's ACL.
    static void owner_only(const std::filesystem::path& d) {
#if !defined(_WIN32)
        std::filesystem::permissions(d, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
#else
        (void)d;
#endif
    }

    static uint64_t round_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
    static uint64_t chunks(uint64_t payload) { return (payload + kChunk - 1) / kChunk; }
    static constexpr char kMagic[16] = "llmx-disk-entry";
    static uint64_t header_bytes(size_t blob, size_t runs, uint64_t chunk_count) {
        return round_up(16 + 4 + 32 + 8 + 4 + 8 * runs + 8 + 4 * chunk_count + blob + 4, kAlign);
    }

    std::filesystem::path path(uint64_t key, const char* ext) const { return dir_ / ("entry-" + std::to_string(key) + ext); }

    bool expired(const std::filesystem::path& file) const {
        if (!options_.max_age) return false;
        std::error_code ec;
        const auto t = std::filesystem::last_write_time(file, ec);
        return ec || std::filesystem::file_time_type::clock::now() - t > std::chrono::seconds(options_.max_age);
    }

    // The header of an entry file, checked: magic, version, `identity` and its own checksum; nothing when any fails.
    struct Header {
        std::string blob;
        std::vector<size_t> run_bytes;
        std::vector<uint32_t> crcs;
        uint64_t size = 0;   // the header's bytes on disk
    };
    std::optional<Header> read_header(format::FileReader& r, const std::array<uint8_t, 32>& identity) const {
        core::HostPages first(kAlign);
        if (r.read(0, first.data(), kAlign) != kAlign) return std::nullopt;
        const uint8_t* p = first.data();
        if (std::memcmp(p, kMagic, 16) != 0) return std::nullopt;
        uint32_t version, runs;
        uint64_t blob, chunk_count;
        std::memcpy(&version, p + 16, 4);
        if (version != kVersion || std::memcmp(p + 20, identity.data(), 32) != 0) return std::nullopt;
        std::memcpy(&blob, p + 52, 8);
        std::memcpy(&runs, p + 60, 4);
        if (runs > 4096) return std::nullopt;
        std::memcpy(&chunk_count, p + 64 + 8 * (size_t)runs, 8);
        if (chunk_count > (uint64_t(1) << 32) || blob > (uint64_t(1) << 32)) return std::nullopt;
        Header h;
        h.size = header_bytes(blob, runs, chunk_count);
        core::HostPages all(h.size);
        std::memcpy(all.data(), first.data(), kAlign);
        if (h.size > kAlign && r.read(kAlign, all.data() + kAlign, h.size - kAlign) != h.size - kAlign) return std::nullopt;
        const uint8_t* q = all.data();
        const size_t end = 16 + 4 + 32 + 8 + 4 + 8 * (size_t)runs + 8 + 4 * chunk_count + blob;
        uint32_t crc;
        std::memcpy(&crc, q + end, 4);
        if (core::crc32c(0, q, end) != crc) return std::nullopt;
        h.run_bytes.resize(runs);
        for (uint32_t i = 0; i < runs; ++i) {
            uint64_t b;
            std::memcpy(&b, q + 64 + 8 * i, 8);
            h.run_bytes[i] = (size_t)b;
        }
        h.crcs.resize(chunk_count);
        std::memcpy(h.crcs.data(), q + 72 + 8 * (size_t)runs, 4 * chunk_count);
        h.blob.assign((const char*)q + 72 + 8 * (size_t)runs + 4 * chunk_count, blob);
        uint64_t payload = 0;
        for (size_t b : h.run_bytes) payload += b;
        if (chunks(payload) != chunk_count) return std::nullopt;
        return h;
    }

    // At start under keep: directory `other`'s entries that are this store's moved into its own under new keys, their last use kept, another build's unproved where `Options::proved` asks (docs/DISK-TIER.md, Adopting another build's entries).
    // True where an entry could not be settled or moved now, a failed read or write and not a bad entry, so the directory is left for a later start.
    bool adopt_from(const std::filesystem::path& other) {
        std::error_code ec;
        const auto text_of = [](const std::filesystem::path& f) {
            std::ifstream in(f, std::ios::binary);
            return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        };
        const auto parts = [](const std::string& text) {
            std::map<std::string, std::string> m;
            for (size_t at = 0; at < text.size();) {
                const size_t end = std::min(text.find('\n', at), text.size()), colon = text.find(": ", at);
                if (colon < end) m[text.substr(at, colon - at)] = text.substr(colon + 2, end - colon - 2);
                at = end + 1;
            }
            return m;
        };
        // The components of `text` that differ from this store's, and whether they are the build's alone.
        const auto differing = [&](const std::string& text, bool& build_only) {
            static const char* const kBuild[] = {"numerics", "compiler", "flags", "shaders", "libm"};
            const auto mine = parts(options_.identity), theirs = parts(text);
            std::string names;
            build_only = mine.size() == theirs.size();
            for (const auto& part : mine) {
                const auto it = theirs.find(part.first);
                if (it != theirs.end() && it->second == part.second) continue;
                names += (names.empty() ? "" : ", ") + part.first;
                build_only = build_only && std::find_if(std::begin(kBuild), std::end(kBuild), [&](const char* n) { return part.first == n; }) != std::end(kBuild);
            }
            return names;
        };
        const std::string theirs = text_of(other / "identity");
        bool build_only = false, retain = false;
        size_t unproved = 0;
        if (!options_.identity.empty() && theirs.empty())
            for (const auto& e : std::filesystem::directory_iterator(other, ec))
                if (e.path().extension() == ".kv") {
                    refused_.push_back("written by a build that kept no identity beside its entries");
                    break;
                }
        for (const auto& e : std::filesystem::directory_iterator(other, ec)) {
            // A file's own age decides nothing here: a file stands below those used since, which only the index of what the files hold can see (DiskTier::adopt).
            if (e.path().extension() != ".kv") continue;
            // The identity the file was written under: its directory's, or the one kept beside it since an earlier adoption left it unproved.
            std::filesystem::path beside = e.path();
            beside.replace_extension(".from");
            const bool marked = std::filesystem::exists(beside, ec);
            std::string from = theirs;
            if (marked) {
                // A claim may have been cut short: the checksum says whether the header is whole under the identity kept beside it or under its directory's, the claimant's, which then wrote it.
                const std::string kept = text_of(beside);
                const Owner owner = settle(e.path(), identity_of(kept), identity_of(theirs));
                if (owner == Owner::failed) retain = true;
                if (owner == Owner::failed || owner == Owner::none) continue;
                if (owner == Owner::first) from = kept;
            }
            // Then this store's policy for the build that wrote it: its own, another's taken unproved where asked, or not taken.
            const bool foreign_now = !options_.identity.empty() && !from.empty() && from != options_.identity;
            if (foreign_now) {
                differing(from, build_only);
                if (!options_.proved || !build_only) continue;
            }
            const std::array<uint8_t, 32> written = foreign_now ? identity_of(from) : identity_;
            std::optional<Header> h;
            try {
                format::FileReader r(e.path().u8string());
                h = read_header(r, written);
            } catch (const std::exception&) {}
            if (!h) continue;
            const uint64_t key = ++keys_;
            const auto used = std::filesystem::last_write_time(e.path(), ec);
            // The identity an unproved entry was written under is on the disk here before the entry moves, so no failure leaves the entry without it.
            if (foreign_now && !keep_from(key, from)) {
                retain = true;
                continue;
            }
            std::filesystem::rename(e.path(), path(key, ".kv"), ec);
            if (ec) {
                std::filesystem::remove(path(key, ".from"), ec);
                retain = true;
                continue;
            }
            if (foreign_now) {
                foreign_[key] = written;
                ++unproved;
            }
            if (marked) std::filesystem::remove(beside, ec);
            const uint64_t bytes = std::filesystem::file_size(path(key, ".kv"), ec);
            sizes_[key] = bytes;
            uint64_t tag = 0;
            if (foreign_now) std::memcpy(&tag, written.data(), sizeof tag);
            adopted_.push_back(Adopted{key, std::move(h->blob), std::move(h->run_bytes), bytes, used, foreign_now, tag | (foreign_now ? 1 : 0)});
        }
        // A directory of another identity that gave nothing: what differs, for the server's start line.
        if (!options_.identity.empty() && !unproved && !theirs.empty() && theirs != options_.identity)
            refused_.push_back("its " + differing(theirs, build_only) + " differing from this server's");
        return retain;
    }
    // The identity entry `key` was written under, in a file beside where the entry will be, written and on the disk; false where that failed, nothing left.
    bool keep_from(uint64_t key, const std::string& from) const {
        const std::filesystem::path file = path(key, ".from");
        bool ok = !(options_.fault && options_.fault("provenance"));
        if (ok) {
            std::ofstream out(file, std::ios::binary);
            out << from;
            out.flush();
            ok = out.good();
        }
        ok = ok && sync_file(file);
        std::error_code ec;
        if (!ok) std::filesystem::remove(file, ec);
        return ok;
    }

    // Direct writes where the file system takes them and a short probe finds them no slower than writes through the file cache, which on some file systems write several times faster.
    void probe() {
        const size_t bytes = size_t(32) << 20;
        core::HostPages data(bytes);
        std::memset(data.data(), 0x5a, bytes);
        const auto timed = [&](bool direct) -> std::optional<double> {
            const auto file = (dir_ / "probe.tmp").u8string();
            try {
                const auto start = std::chrono::steady_clock::now();
                {
                    format::FileWriter w(file, direct);
                    if (direct && kAlign % w.granule()) throw format::DirectUnavailable(file);
                    w.write(0, data.data(), bytes);
                    w.sync();
                }
                std::error_code ec;
                std::filesystem::remove(std::filesystem::u8path(file), ec);
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            } catch (const std::exception&) {
                std::error_code ec;
                std::filesystem::remove(std::filesystem::u8path(file), ec);
                return std::nullopt;
            }
        };
        const std::optional<double> buffered = timed(false), direct = timed(true);
        direct_ = direct && (!buffered || *direct <= *buffered);
        const std::optional<double> taken = direct_ ? direct : buffered;
        if (taken && *taken > 0) rate_.store((double)bytes / *taken);
    }

    // The I/O thread: reads first, then writes, each in the order queued; once a ten-minute wait passes with nothing queued, a sweep.
    void run() {
        auto last_sweep = std::chrono::steady_clock::now();
        for (;;) {
            std::shared_ptr<Job> j;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait_for(lk, std::chrono::seconds(60), [&] { return stopping_ || !jobs_.empty(); });
                if (jobs_.empty()) {
                    if (stopping_) return;
                    if (std::chrono::steady_clock::now() - last_sweep >= std::chrono::minutes(10)) {
                        lk.unlock();
                        sweep();
                        last_sweep = std::chrono::steady_clock::now();
                    }
                    continue;
                }
                auto it = std::find_if(jobs_.begin(), jobs_.end(), [](const std::shared_ptr<Job>& x) { return !x->put; });
                if (it == jobs_.end()) it = jobs_.begin();
                j = *it;
                jobs_.erase(it);
                running_ = j;
            }
            std::string error;
            bool ok = false;
            try {
                ok = j->put ? write(*j, error) : read(*j, error);
            } catch (const std::exception& e) {
                error = e.what();
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                running_.reset();
            }
            if (j->done) j->done(ok, error);
        }
    }

    // Reads queued while a write runs, each served between two of its chunks, so a read waits for one chunk of a write rather than the whole of it; the staging chunk is free there, its bytes written.
    void serve_reads() {
        for (;;) {
            std::shared_ptr<Job> r;
            {
                std::lock_guard<std::mutex> lk(m_);
                const auto it = std::find_if(jobs_.begin(), jobs_.end(), [](const std::shared_ptr<Job>& x) { return !x->put; });
                if (it == jobs_.end()) return;
                r = *it;
                jobs_.erase(it);
            }
            std::string error;
            bool ok = false;
            try {
                ok = read(*r, error);
            } catch (const std::exception& e) {
                error = e.what();
            }
            if (r->done) r->done(ok, error);
        }
    }

    // The payload as one stream over the runs' slabs: copy `n` bytes at payload offset `at` to or from `buf`.
    static void stream(const Job& j, uint64_t at, uint8_t* buf, size_t n, bool to_slabs) {
        size_t run = 0;
        while (run < j.runs.size() && at >= j.runs[run].bytes) at -= j.runs[run++].bytes;
        while (n) {
            const StoreRun& r = j.runs[run];
            const size_t in = (r.offset + (size_t)at) % j.slab, k = std::min({n, j.slab - in, r.bytes - (size_t)at});
            auto* slab = (uint8_t*)const_cast<void*>(r.slabs[(r.offset + (size_t)at) / j.slab]->host_ptr());
            if (to_slabs) std::memcpy(slab + in, buf, k);
            else std::memcpy(buf, slab + in, k);
            buf += k;
            n -= k;
            at += k;
            if (at == r.bytes) {
                at = 0;
                ++run;
            }
        }
    }

    bool write(Job& j, std::string& error) {
        const std::vector<size_t>& run_bytes = j.layout;
        uint64_t payload = 0;
        for (size_t b : run_bytes) payload += b;
        const uint64_t chunk_count = chunks(payload), head = header_bytes(j.blob.size(), run_bytes.size(), chunk_count), total = head + round_up(payload, kAlign);
        const std::optional<uint64_t> free = free_bytes();
        if (free && (*free < total || *free - total < options_.floor)) {
            error = "the file system would keep less than the free-space floor";
            return false;
        }
        const auto tmp = path(j.key, ".tmp");
        std::vector<uint32_t> crcs(chunk_count);
        const auto began = std::chrono::steady_clock::now();
        try {
            format::FileWriter w(tmp.u8string(), direct_);
            for (uint64_t c = 0; c < chunk_count; ++c) {
                if (j.cancelled) throw std::runtime_error("cancelled");
                const size_t n = (size_t)std::min<uint64_t>(kChunk, payload - c * kChunk), padded = (size_t)round_up(n, kAlign);
                stream(j, c * kChunk, staging_.data(), n, false);
                std::memset(staging_.data() + n, 0, padded - n);
                crcs[c] = core::crc32c(0, staging_.data(), n);
                w.write(head + c * kChunk, staging_.data(), padded);
                serve_reads();
                if (options_.pace.count()) {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_.wait_for(lk, options_.pace, [&] { return j.cancelled.load() || stopping_; });
                }
            }
            core::HostPages h(head);
            std::memset(h.data(), 0, head);
            uint8_t* p = h.data();
            const uint32_t version = kVersion, runs = (uint32_t)run_bytes.size();
            const uint64_t blob = j.blob.size();
            std::memcpy(p, kMagic, 16);
            std::memcpy(p + 16, &version, 4);
            std::memcpy(p + 20, identity_.data(), 32);
            std::memcpy(p + 52, &blob, 8);
            std::memcpy(p + 60, &runs, 4);
            for (size_t i = 0; i < run_bytes.size(); ++i) {
                const uint64_t b = run_bytes[i];
                std::memcpy(p + 64 + 8 * i, &b, 8);
            }
            std::memcpy(p + 64 + 8 * run_bytes.size(), &chunk_count, 8);
            std::memcpy(p + 72 + 8 * run_bytes.size(), crcs.data(), 4 * chunk_count);
            std::memcpy(p + 72 + 8 * run_bytes.size() + 4 * chunk_count, j.blob.data(), blob);
            const size_t end = 72 + 8 * run_bytes.size() + 4 * chunk_count + blob;
            const uint32_t crc = core::crc32c(0, p, end);
            std::memcpy(p + end, &crc, 4);
            w.write(0, p, head);
            w.sync();
            w.close();
            if (j.cancelled) throw std::runtime_error("cancelled");
            std::filesystem::rename(tmp, path(j.key, ".kv"));
        } catch (const std::exception& e) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            error = e.what();
            return false;
        }
        sync_directory();
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        if (took > 0 && !options_.pace.count()) rate_.store(rate_.load() > 0 ? 0.5 * rate_.load() + 0.5 * (double)total / took : (double)total / took);
        std::lock_guard<std::mutex> lk(m_);
        // An evict that came while the file was renamed into place wins.
        if (j.cancelled) {
            std::error_code ec;
            std::filesystem::remove(path(j.key, ".kv"), ec);
            error = "cancelled";
            return false;
        }
        sizes_[j.key] = total;
        return true;
    }

    bool read(Job& j, std::string& error) {
        const auto file = path(j.key, ".kv");
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!sizes_.count(j.key)) {
                error = "no such entry";
                return false;
            }
        }
        // Why the entry cannot be read, found with the file open; the entry is deleted once the file is closed, as some systems refuse to remove an open file.
        const std::string why = [&]() -> std::string {
            try {
                std::unique_ptr<format::FileReader> r;
                try {
                    r = std::make_unique<format::FileReader>(file.u8string(), true);
                    if (kAlign % r->granule()) r.reset();
                } catch (const format::DirectUnavailable&) {}
                if (!r) r = std::make_unique<format::FileReader>(file.u8string(), false);
                const std::optional<Header> h = read_header(*r, written_as(j.key));
                if (!h) return "an entry whose header is not this server's or fails its checksum";
                if (j.claim) return "";
                uint64_t payload = 0;
                if (h->run_bytes != j.layout) return "an entry of another layout";
                for (size_t b : h->run_bytes) payload += b;
                for (uint64_t c = 0; c < h->crcs.size(); ++c) {
                    const size_t n = (size_t)std::min<uint64_t>(kChunk, payload - c * kChunk), padded = (size_t)round_up(n, kAlign);
                    if (r->read(h->size + c * kChunk, staging_.data(), padded) < n) return "an entry shorter than its header says";
                    if (core::crc32c(0, staging_.data(), n) != h->crcs[c]) return "an entry that fails its checksum";
                    stream(j, c * kChunk, staging_.data(), n, true);
                    if (options_.pace.count()) {
                        std::unique_lock<std::mutex> lk(m_);
                        cv_.wait_for(lk, options_.pace, [&] { return stopping_; });
                    }
                }
                return "";
            } catch (const std::exception& e) {
                return std::string("an entry not read: ") + e.what();
            }
        }();
        std::string failed = why;
        const std::array<uint8_t, 32> theirs = written_as(j.key);
        if (failed.empty() && j.claim && !rewrite(file, j.key)) {
            // A proof held and a write did not: the entry stays under whichever identity its header is whole, the one beside it kept for the next start, and goes only where it is whole under neither.
            const Owner owner = settle(file, theirs, identity_);
            if (owner == Owner::none) failed = "an entry a claim left whole under neither build";
            else {
                std::lock_guard<std::mutex> lk(m_);
                if (owner == Owner::second) foreign_.erase(j.key);
                error = "a claim that did not reach the disk; the entry is kept and proved again after a restart";
                return true;
            }
        }
        if (failed.empty()) return true;
        std::lock_guard<std::mutex> lk(m_);
        sizes_.erase(j.key);
        foreign_.erase(j.key);
        std::error_code ec;
        std::filesystem::remove(file, ec);
        std::filesystem::remove(path(j.key, ".from"), ec);
        error = failed;
        return false;
    }

    // The identity entry `key`'s header carries: this store's, or for an unproved entry the one it was written under.
    std::array<uint8_t, 32> written_as(uint64_t key) const {
        std::lock_guard<std::mutex> lk(m_);
        const auto it = foreign_.find(key);
        return it == foreign_.end() ? identity_ : it->second;
    }

    // The bytes of an entry file's header up to its checksum, and where the checksum lies; empty where the file is too short to hold one.
    static std::vector<uint8_t> header_of(std::fstream& f, size_t& end) {
        std::vector<uint8_t> head(72);
        f.seekg(0);
        if (!f.read((char*)head.data(), 72) || std::memcmp(head.data(), kMagic, 16) != 0) return {};
        uint64_t blob, chunk_count;
        uint32_t runs;
        std::memcpy(&blob, head.data() + 52, 8);
        std::memcpy(&runs, head.data() + 60, 4);
        if (runs > 4096 || blob > (uint64_t(1) << 32)) return {};
        head.resize(72 + 8 * (size_t)runs);
        f.seekg(0);
        if (!f.read((char*)head.data(), (std::streamsize)head.size())) return {};
        std::memcpy(&chunk_count, head.data() + 64 + 8 * (size_t)runs, 8);
        if (chunk_count > (uint64_t(1) << 32)) return {};
        end = 16 + 4 + 32 + 8 + 4 + 8 * (size_t)runs + 8 + 4 * (size_t)chunk_count + (size_t)blob;
        head.resize(end + 4);
        f.seekg(0);
        if (!f.read((char*)head.data(), (std::streamsize)head.size())) return {};
        return head;
    }

    // Which of two identities an entry's header is whole under, whatever a claim cut short left in its identity field; `failed` where the file could not be read or put right now, which says nothing of the entry.
    // A claim changes the field and the checksum alone, so the checksum on disk settles which header it belongs to, and the field is written to match it.
    enum class Owner { none, first, second, failed };
    Owner settle(const std::filesystem::path& file, const std::array<uint8_t, 32>& first, const std::array<uint8_t, 32>& second) const {
        std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
        if (!f) return Owner::failed;
        size_t end = 0;
        std::vector<uint8_t> head = header_of(f, end);
        if (head.empty()) return f.bad() ? Owner::failed : Owner::none;
        uint32_t crc;
        std::memcpy(&crc, head.data() + end, 4);
        for (const Owner which : {Owner::first, Owner::second}) {
            const std::array<uint8_t, 32>& id = which == Owner::first ? first : second;
            const bool as_written = std::memcmp(head.data() + 20, id.data(), 32) == 0;
            std::vector<uint8_t> tried = head;
            std::memcpy(tried.data() + 20, id.data(), 32);
            if (core::crc32c(0, tried.data(), end) != crc) continue;
            if (as_written) return which;
            if (options_.fault && options_.fault("repair")) return Owner::failed;
            f.clear();
            f.seekp(20);
            if (!f.write((const char*)id.data(), 32) || !f.flush()) return Owner::failed;
            f.close();
            return sync_file(file) ? which : Owner::failed;
        }
        return Owner::none;
    }

    // An unproved entry's header rewritten under this store's identity, with its checksum, and on the disk before the identity kept beside it goes; the payload is untouched.
    // The write is in place and through no layer of src/format, which holds no in-place write and is among the fingerprinted sources.
    bool rewrite(const std::filesystem::path& file, uint64_t key) {
        const std::array<uint8_t, 32> theirs = written_as(key);
        if (theirs == identity_) return true;
        {
            std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
            size_t end = 0;
            std::vector<uint8_t> head = header_of(f, end);
            if (head.empty() || std::memcmp(head.data() + 20, theirs.data(), 32) != 0) return false;
            std::memcpy(head.data() + 20, identity_.data(), 32);
            const uint32_t crc = core::crc32c(0, head.data(), end);
            std::memcpy(head.data() + end, &crc, 4);
            f.seekp(0);
            if (!f.write((const char*)head.data(), (std::streamsize)head.size()) || !f.flush()) return false;
        }
        if ((options_.fault && options_.fault("claim-sync")) || !sync_file(file)) return false;
        std::error_code ec;
        std::filesystem::remove(path(key, ".from"), ec);
        sync_directory();
        std::lock_guard<std::mutex> lk(m_);
        foreign_.erase(key);
        return true;
    }

    // A file's bytes on the disk, not only with the operating system.
    static bool sync_file(const std::filesystem::path& file) {
#if defined(_WIN32)
        const HANDLE h = CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        const bool ok = FlushFileBuffers(h) != 0;
        CloseHandle(h);
        return ok;
#else
        const int fd = ::open(file.c_str(), O_WRONLY | O_CLOEXEC);
        if (fd < 0) return false;
        const bool ok = fsync(fd) == 0;
        ::close(fd);
        return ok;
#endif
    }

    // A rename reaches the disk with its directory.
    void sync_directory() const {
#if !defined(_WIN32)
        const int fd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd >= 0) {
            fsync(fd);
            ::close(fd);
        }
#endif
    }

    Options options_;
    std::atomic<double> rate_{0};
    std::array<uint8_t, 32> identity_;
    std::filesystem::path dir_;
    Lock lock_;
    bool direct_ = false;
    core::HostPages staging_;              // the I/O thread's page-aligned buffer, one chunk
    std::vector<std::string> refused_;
    std::vector<Adopted> adopted_;
    std::map<uint64_t, std::array<uint8_t, 32>> foreign_;   // under m_, the unproved entries and the identity each was written under
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> jobs_;  // under m_
    std::shared_ptr<Job> running_;           // under m_
    std::map<uint64_t, uint64_t> sizes_;     // under m_, the entries in place and their files' bytes
    uint64_t keys_ = 0;                      // under m_
    bool stopping_ = false;                  // under m_
    std::thread thread_;
};

} // namespace server
