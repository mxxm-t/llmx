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

// One device's run of an entry, as host memory holds it: `bytes` over consecutive host-visible slabs of the store call's slab size (infer::HostHistory).
struct StoreRun {
    std::vector<backend::BufferPtr> slabs;
    size_t bytes = 0;
};

class DiskStore {
public:
    struct Options {
        std::string root;            // the directory every server's own directory sits under (--disk-cache-dir)
        uint64_t floor = 0;          // free space the file system keeps after every write (--disk-cache-floor)
        bool keep = false;           // at a clean exit leave the entries for the next server, and at start adopt those other servers left (--disk-cache-keep)
        uint64_t max_age = 0;        // seconds after an entry's last use when sweeps and adoption delete it, 0 for never (--disk-cache-max-age)
        std::chrono::milliseconds pace{0};   // a pause after each chunk written or read, a write's cut short by a cancel, which tests hold a call in flight with
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
    };

    // The entries' alignment on disk, which every direct-I/O granule up to it divides, and the unit a checksum covers.
    static constexpr size_t kAlign = size_t(1) << 20, kChunk = size_t(4) << 20;

    // Makes the server's own directory under the root, locked for the store's life, after sweeping the root (sweep) and, with keep, adopting what other servers left whose identity is `identity`; then chooses the write mode by a short probe and starts the I/O thread.
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

    // Queues an entry's write and returns its key; `done` runs on the I/O thread once the file is in place, or with the reason it is not: the floor, a failed write, or a cancel.
    // The slabs must not change until `done` has run.
    uint64_t put(std::string blob, std::vector<StoreRun> runs, size_t slab, Done done) {
        auto j = std::make_shared<Job>();
        j->put = true;
        j->blob = std::move(blob);
        j->runs = std::move(runs);
        j->slab = slab;
        j->done = std::move(done);
        std::lock_guard<std::mutex> lk(m_);
        j->key = ++keys_;
        jobs_.push_back(j);
        cv_.notify_all();
        return j->key;
    }

    // Queues a read of entry `key` into the runs' slabs, whose bytes must be the entry's; reads go before writes. `done` reports a missing entry, another identity or layout, a failed checksum or read, after which the entry is deleted.
    void get(uint64_t key, std::vector<StoreRun> runs, size_t slab, Done done) {
        auto j = std::make_shared<Job>();
        j->key = key;
        j->runs = std::move(runs);
        j->slab = slab;
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

    // Removes every directory under the root whose lock is free, which a server that ended left, but a kept one younger than the age limit; with `adopt`, at start under keep, adopts first the entries of those whose identity is this store's, younger than the age limit.
    // A directory whose lock is held is a live server's and is never touched.
    void sweep(bool adopt = false) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(std::filesystem::u8path(options_.root), ec)) {
            const auto other = e.path();
            if (other == dir_ || !e.is_directory(ec) || other.filename().u8string().rfind("server-", 0) != 0) continue;
            Lock held = lock(other);
            if (!held.held()) continue;
            const bool kept = std::filesystem::exists(other / "kept", ec);
            if (adopt && options_.keep) adopt_from(other);
            else if (kept && !expired(other / "kept")) continue;
            held = Lock{};
            std::filesystem::remove_all(other, ec);
        }
    }

private:
    struct Job {
        bool put = false;
        uint64_t key = 0;
        std::string blob;
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
    static constexpr uint32_t kVersion = 1;
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

    // The header of an entry file, checked: magic, version, identity and its own checksum; nothing when any fails.
    struct Header {
        std::string blob;
        std::vector<size_t> run_bytes;
        std::vector<uint32_t> crcs;
        uint64_t size = 0;   // the header's bytes on disk
    };
    std::optional<Header> read_header(format::FileReader& r) const {
        core::HostPages first(kAlign);
        if (r.read(0, first.data(), kAlign) != kAlign) return std::nullopt;
        const uint8_t* p = first.data();
        if (std::memcmp(p, kMagic, 16) != 0) return std::nullopt;
        uint32_t version, runs;
        uint64_t blob, chunk_count;
        std::memcpy(&version, p + 16, 4);
        if (version != kVersion || std::memcmp(p + 20, identity_.data(), 32) != 0) return std::nullopt;
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

    // At start under keep: every entry of directory `other` with this store's identity, younger than the age limit, moved into this store's directory under a key of its own.
    void adopt_from(const std::filesystem::path& other) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(other, ec)) {
            if (e.path().extension() != ".kv" || expired(e.path())) continue;
            std::optional<Header> h;
            try {
                format::FileReader r(e.path().u8string());
                h = read_header(r);
            } catch (const std::exception&) {}
            if (!h) continue;
            const uint64_t key = ++keys_;
            const auto used = std::filesystem::last_write_time(e.path(), ec);
            std::filesystem::rename(e.path(), path(key, ".kv"), ec);
            if (ec) continue;
            const uint64_t bytes = std::filesystem::file_size(path(key, ".kv"), ec);
            sizes_[key] = bytes;
            adopted_.push_back(Adopted{key, std::move(h->blob), std::move(h->run_bytes), bytes, used});
        }
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
            const size_t in = (size_t)at % j.slab, k = std::min({n, j.slab - in, r.bytes - (size_t)at});
            auto* slab = (uint8_t*)const_cast<void*>(r.slabs[(size_t)at / j.slab]->host_ptr());
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
        std::vector<size_t> run_bytes;
        uint64_t payload = 0;
        for (const StoreRun& r : j.runs) {
            run_bytes.push_back(r.bytes);
            payload += r.bytes;
        }
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
                const std::optional<Header> h = read_header(*r);
                if (!h) return "an entry whose header is not this server's or fails its checksum";
                uint64_t payload = 0;
                bool layout = h->run_bytes.size() == j.runs.size();
                for (size_t i = 0; layout && i < j.runs.size(); ++i) layout = h->run_bytes[i] == j.runs[i].bytes;
                if (!layout) return "an entry of another layout";
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
        if (why.empty()) return true;
        std::lock_guard<std::mutex> lk(m_);
        sizes_.erase(j.key);
        std::error_code ec;
        std::filesystem::remove(file, ec);
        error = why;
        return false;
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
    std::vector<Adopted> adopted_;
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
