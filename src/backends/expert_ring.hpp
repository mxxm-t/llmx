#pragma once
// The device expert cache's ring and reader (docs/src/backends-expert_ring.md): the one owner, for every GPU backend, of the thread that copies a wave's missing experts into ring entries the device pulls from.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "backends/backend.hpp"
#include "core/job_threads.hpp"
#include "format/file_reader.hpp"

namespace backend {

// A device's ring: an exchange of host-coherent words and 2 * half ring entries a stack, allocated here as Memory::host_coherent, and the reader that fills them.
// The exchange holds a header (kFault, kUnplaced) and two halves that waves alternate by their seq, each a 64-bit ticket, the pulls n, the seq's low word and n pairs {slot << 16 | expert, ring entry}, each part on 64 bytes.
class ExpertRing {
public:
    static constexpr size_t kFault = 0, kUnplaced = 1;
    static constexpr size_t kTicket = 0, kPulls = 2, kTag = 3, kPairs = 4;
    static constexpr size_t kHalf = 8;               // experts a wave, half the ring's entries
    static constexpr size_t kMaxHalf = 64;           // llmx_expert_wave's want list
    static constexpr size_t kThreads = 4;
    static constexpr uint64_t kSeqLimit = uint64_t(1) << 31;
    static constexpr size_t kAlign = 16;  // words a header or half is rounded up to
    static size_t half_words(size_t half) { return (kPairs + 2 * half + kAlign - 1) / kAlign * kAlign; }
    static size_t words(size_t half) { return kAlign + 2 * half_words(half); }

    ExpertRing(Backend& b, size_t half, const size_t entry[3]) : b_(b), half_(half), pool_(kThreads - 1) {
        if (!half || half > kMaxHalf) throw std::runtime_error("expert ring: a wave of " + std::to_string(half) + " experts, outside 1 to 64");
        xchg_ = b.alloc(words(half) * sizeof(uint32_t), Memory::host_coherent);
        exchange_ = (uint32_t*)const_cast<void*>(xchg_->host_ptr());
        for (int i = 0; i < 3; ++i) {
            entry_[i] = entry[i];
            mem_[i] = b.alloc(size_mul(2 * half, entry[i]), Memory::host_coherent);
            ring_[i] = (uint8_t*)const_cast<void*>(mem_[i]->host_ptr());
        }
        reader_ = std::thread([this] { read(); });
    }
    ~ExpertRing() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        reader_.join();
    }
    ExpertRing(const ExpertRing&) = delete;
    ExpertRing& operator=(const ExpertRing&) = delete;

    // A layer's host stacks: each stack's first expert and an expert's bytes, the same in its ring entries, and where given the file read in place of its memory (ExpertSlots::Stack).
    struct Source {
        const uint8_t* at[3];
        size_t stride[3];
        const format::FileReader* file[3];
        uint64_t file_at[3];
        ExpertSlots::RingCounts* counts;
        bool one_row;
    };
    // A cached step's source counts: its layer's counts, and whether it is a one-row step, which it counts.
    static void count_step(Source& s, const ExpertSlots& c, size_t rows) {
        s.counts = c.counts;
        s.one_row = rows == 1;
        if (s.counts && s.one_row) s.counts->one_steps.fetch_add(1, std::memory_order_relaxed);
    }

    // A fault from a pull or a short round, thrown once the device is idle by the backend's next check of its results or by the next cached step.
    // It leaves slots claiming experts whose bytes never came, so every live state the ring served is zeroed first.
    static void raise(const std::unique_ptr<ExpertRing>& ring, Backend& b, const std::string& prefix) {
        if (!ring || !ring->faulted()) return;
        b.sync();
        ring->reset();
        for (const auto& [owner, n] : ring->states_) {
            const BufferPtr s = owner.lock();
            if (!s) continue;
            const std::vector<uint32_t> zero(n, 0);
            b.write(*s, 0, zero.data(), n * sizeof(uint32_t));
        }
        b.sync();
        throw std::runtime_error(prefix + "an expert cache pull found its experts unread, or a step's experts outnumbered what its slots took; the cache starts empty");
    }
    // The device's ring for `c`'s step, made at its first cached step and again for a wider expert, the layers it served kept.
    static ExpertRing& of(std::unique_ptr<ExpertRing>& ring, Backend& b, const ExpertSlots& c, const std::string& prefix) {
        raise(ring, b, prefix);
        // Pulls compare the seq's low word, so the ring starts over long before it wraps.
        if (ring && ring->seq_ >= kSeqLimit) {
            b.sync();
            ring->reset();
        }
        bool fits = ring != nullptr;
        for (int i = 0; i < 3 && fits; ++i) fits = c.stack[i].stride <= ring->entry_[i];
        if (!fits) {
            size_t entry[3];
            std::vector<std::pair<std::weak_ptr<Buffer>, size_t>> states;
            for (int i = 0; i < 3; ++i) entry[i] = std::max(ring ? ring->entry_[i] : 0, c.stack[i].stride);
            if (ring) {
                b.sync();
                states = std::move(ring->states_);
            }
            ring.reset();
            ring = std::make_unique<ExpertRing>(b, kHalf, entry);
            ring->states_ = std::move(states);
        }
        auto& st = ring->states_;
        st.erase(std::remove_if(st.begin(), st.end(), [](const auto& s) { return s.first.expired(); }), st.end());
        if (std::find_if(st.begin(), st.end(), [&](const auto& s) { return s.first.lock().get() == c.state.buffer; }) == st.end()) st.push_back({c.state_owner, c.words()});
        return *ring;
    }

    size_t half() const { return half_; }
    // The exchange and each stack's entries, which a pull binds, and an entry's bytes.
    const Buffer& exchange_buffer() const { return *xchg_; }
    const Buffer& entries(int i) const { return *mem_[i]; }
    size_t entry(int i) const { return entry_[i]; }
    // The next wave's seq, from 1, and the word its half starts at.
    uint64_t next() { return ++seq_; }
    size_t half_at(uint64_t seq) const { return kAlign + (seq & 1) * half_words(half_); }

    // A step's rounds and a round's waves, fixed by the host: rounds of the slots cover every picked expert, and a round's waves every slot it can fill.
    struct Plan {
        size_t rounds, waves;
    };
    Plan plan(size_t entries, size_t n_expert, size_t slots) const {
        const size_t needed = std::min(entries, n_expert);
        return {(needed + slots - 1) / slots, (std::min(slots, needed) + half_ - 1) / half_};
    }
    // A cached step's order, the one every backend records: wave g + 1 before pull g, and round r + 1's first wave after round r's ids and before its products, so the reader's copies overlap device work.
    // wave() records a wave and hands it to request, returning its seq; pull(seq) waits on the ticket and copies; round(r, last) runs llmx_expert_round's ids; products() the round's experts.
    template <class Wave, class Pull, class Round, class Products>
    static void run(const Plan& p, Wave&& wave, Pull&& pull, Round&& round, Products&& products) {
        uint64_t cur = wave();
        for (size_t r = 0; r < p.rounds; ++r) {
            for (size_t w = 0; w < p.waves; ++w) {
                const uint64_t next = w + 1 < p.waves ? wave() : 0;
                pull(cur);
                cur = next;
            }
            round(r, r + 1 == p.rounds);
            if (r + 1 < p.rounds) cur = wave();
            products();
        }
    }

    // After the recording thread has submitted wave `seq` under ticket t: the reader waits for t, then fills the wave's pulls and writes its ticket.
    void request(Ticket t, uint64_t seq, const Source& s) {
        {
            std::lock_guard<std::mutex> lock(m_);
            queue_.push_back({t, seq, s});
        }
        cv_.notify_one();
    }
    // Whether a pull found its ticket unfilled or a last round left a picked expert outside a slot, read from the host's copy of the header.
    bool faulted() const { return load(exchange_ + kFault) || load(exchange_ + kUnplaced); }
    // Once the device has retired every requested wave: the reader idle, so it touches no layer's counts or host stacks after their model goes.
    void drain() {
        std::unique_lock<std::mutex> lock(m_);
        idle_.wait(lock, [this] { return queue_.empty() && !busy_; });
    }
    // Once the device is idle and the queue drained: tickets and header back to zero and seq from 1 again, so the caller can zero its slot states and start over.
    void reset() {
        drain();
        std::memset(exchange_, 0, words(half_) * sizeof(uint32_t));
        std::atomic_thread_fence(std::memory_order_seq_cst);
        seq_ = 0;
    }

private:
    struct Request {
        Ticket t;
        uint64_t seq;
        Source s;
    };
    static uint64_t ns(std::chrono::steady_clock::duration d) { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(d).count(); }
    static uint32_t load(const uint32_t* w) { return reinterpret_cast<const std::atomic<uint32_t>*>(w)->load(std::memory_order_acquire); }

    void read() {
        for (;;) {
            Request r;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                r = queue_.front();
                queue_.pop_front();
                busy_ = true;
            }
            fill(r);
            {
                std::lock_guard<std::mutex> lock(m_);
                busy_ = false;
            }
            idle_.notify_all();
        }
    }
    // The reader's only backend call is the wait; a failed submission retires without its work, so a half not tagged with the seq is left alone (backend.hpp, Backend::wait).
    void fill(const Request& r) {
        const auto t0 = std::chrono::steady_clock::now();
        b_.wait(r.t);
        const auto t1 = std::chrono::steady_clock::now();
        uint32_t* w = exchange_ + half_at(r.seq);
        if (load(w + kTag) != (uint32_t)r.seq) return;
        const size_t n = load(w + kPulls), base = (r.seq & 1) * half_;
        if (ExpertSlots::RingCounts* c = r.s.counts) {
            c->waves.fetch_add(1, std::memory_order_relaxed);
            c->wait_ns.fetch_add(ns(t1 - t0), std::memory_order_relaxed);
        }
        if (!n) return;
        bool failed = n > half_;
        if (!failed) {
            try {
                size_t pieces[3];
                for (int i = 0; i < 3; ++i) pieces[i] = (r.s.stride[i] + ExpertSlots::kPiece - 1) / ExpertSlots::kPiece;
                const size_t per = pieces[0] + pieces[1] + pieces[2];
                pool_.run(n * per, [&](size_t job) {
                    const size_t p = job / per;
                    size_t q = job % per;
                    int i = 0;
                    while (q >= pieces[i]) q -= pieces[i++];
                    const size_t e = load(w + kPairs + 2 * p) & 0xffffu, stride = r.s.stride[i], at = q * ExpertSlots::kPiece, len = std::min(ExpertSlots::kPiece, stride - at);
                    uint8_t* to = ring_[i] + (base + p) * stride + at;
                    if (!r.s.file[i]) std::memcpy(to, r.s.at[i] + e * stride + at, len);
                    else if (r.s.file[i]->read(r.s.file_at[i] + e * stride + at, to, len) != len) throw std::runtime_error("expert ring: a short read of an expert");
                });
                for (size_t p = 0; p < n; ++p) w[kPairs + 2 * p + 1] = uint32_t(base + p);
            } catch (const std::exception&) {
                failed = true;
            }
        }
        if (ExpertSlots::RingCounts* c = r.s.counts) {
            const uint64_t t = ns(std::chrono::steady_clock::now() - t1);
            c->read_ns.fetch_add(t, std::memory_order_relaxed);
            if (!failed) c->bytes.fetch_add(n * (r.s.stride[0] + r.s.stride[1] + r.s.stride[2]), std::memory_order_relaxed);
            if (r.s.one_row) {
                c->one_pulled.fetch_add(n, std::memory_order_relaxed);
                c->one_read_ns.fetch_add(t, std::memory_order_relaxed);
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        reinterpret_cast<std::atomic<uint64_t>*>(w + kTicket)->store(r.seq | (failed ? uint64_t(1) << 63 : 0), std::memory_order_release);
    }

    Backend& b_;
    const size_t half_;
    BufferPtr xchg_, mem_[3];
    size_t entry_[3];
    uint32_t* exchange_;
    uint8_t* ring_[3];
    std::vector<std::pair<std::weak_ptr<Buffer>, size_t>> states_;  // the layers' states the ring served and their words, zeroed on a fault while they live
    uint64_t seq_ = 0;
    core::JobThreads pool_;
    std::mutex m_;
    std::condition_variable cv_, idle_;
    std::deque<Request> queue_;
    bool stop_ = false, busy_ = false;
    std::thread reader_;
};

}  // namespace backend
