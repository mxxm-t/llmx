#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <exception>
#include <limits>
#include <vector>
#include <atomic>
#include <cstdlib>

#include "backends/backend.hpp"
#include "backends/cpu/prefill_placement.hpp"
#include "backends/cpu/q8_dots.hpp"
#include "core/fp16.hpp"
#include "core/host_memory.hpp"
#include "format/gguf.hpp"
#include "quant/quant.hpp"

// Platform-agnostic intrinsics headers. MSVC uses <intrin.h> for __cpuid;
// GCC/Clang use <cpuid.h> (for __get_cpuid) and <immintrin.h> for AVX2.
// <immintrin.h> exists on all three; only the cpuid call differs.
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <immintrin.h>

namespace backend {

// Host storage.
// `adopt` keeps the caller's pointer, which is the whole point: the weights are already resident in the GGUF payload and copying an 8 GB model to make it a buffer would double peak memory for nothing.
// `alloc` owns its bytes instead.
class CpuBuffer final : public Buffer {
public:
    explicit CpuBuffer(size_t bytes) : owned_(bytes), size_(bytes) { ptr_ = owned_.data(); }
    CpuBuffer(const void* adopted, size_t bytes)
        : size_(bytes), ptr_(const_cast<void*>(adopted)) {}

    size_t size() const override { return size_; }
    const void* host_ptr() const override { return ptr_; }
    void* host_address() const { return ptr_; }

private:
    std::vector<uint8_t> owned_;
    size_t size_ = 0;
    void* ptr_ = nullptr;
};

// 128 tokens per KV block: chosen by the real-model screening recorded in docs/KV-CACHE.md. 64 lost prefill consistently; 256 was not separable from 128 on decode and doubles the partial-tail waste.
static const size_t KV_BLOCK_TOKENS = 128;

// KV blocks, one K and one V buffer per layer; block b starts at b*block_floats() and holds [kv_head][token][head_dim].
// Blocks are backed in doubling steps as ids are first written.
// The storage allocates through its backend, so growth is alloc then copy.
// A resolved host pointer per layer is cached since attention asks for one per head, query and block.
class CpuKVStorage final : public KVStorage {
public:
    CpuKVStorage(Backend& owner, size_t layers, size_t heads, size_t dim,
                 size_t max_blocks, KVType kt, KVType vt)
        : owner_(&owner), heads_(heads), dim_(dim), max_(max_blocks), kt_(kt), vt_(vt),
          kb_(kv_elem_bytes(kt)), vb_(kv_elem_bytes(vt)),
          k_(layers), v_(layers), kp_(layers, nullptr), vp_(layers, nullptr) {
        // Called for its overflow throw, not its value: block_floats() recomputes this on every access and must not wrap.
        mul(mul(heads, KV_BLOCK_TOKENS), dim);
    }

    static size_t mul(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a)
            throw std::runtime_error("backend: KV storage size overflows");
        return a * b;
    }
    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a)
            throw std::runtime_error("backend: KV storage size overflows");
        return a + b;
    }
    // Whole blocks for `tokens` positions, without the usual +bt-1 overflow.
    static size_t blocks_for(size_t tokens) {
        return tokens / KV_BLOCK_TOKENS + (tokens % KV_BLOCK_TOKENS != 0);
    }

    size_t max_blocks() const override { return max_; }
    size_t allocated_bytes() const override {
        size_t bytes = 0;
        for (size_t l = 0; l < k_.size(); ++l)
            if (k_[l]) bytes += k_[l]->size() + v_[l]->size();
        return bytes;
    }
    size_t peak_bytes() const override { return peak_; }
    size_t layers() const { return k_.size(); }
    size_t heads() const { return heads_; }
    size_t dim() const { return dim_; }
    size_t block_floats() const { return heads_ * KV_BLOCK_TOKENS * dim_; }
    KVType k_type() const { return kt_; }
    KVType v_type() const { return vt_; }
    size_t k_block_bytes() const { return block_floats() * kb_; }
    size_t v_block_bytes() const { return block_floats() * vb_; }
    bool backed(size_t id) const { return id < backed_; }

    // The complete new set is allocated and already holds the history before any of it is published, so an allocation that throws leaves the storage exactly as it was and a retry starts over. alloc is zero-filled, which is what leaves a newly backed block reading as zeros.
    void ensure(size_t id) {
        if (id < backed_) return;
        if (id >= max_) throw std::runtime_error("backend: KV block outside the budget");
        const size_t want = std::max(id + 1, std::min(max_, backed_ * 2));
        const size_t kbytes = mul(want, k_block_bytes()), vbytes = mul(want, v_block_bytes());
        const size_t held = mul(add(kbytes, vbytes), k_.size());
        std::vector<BufferPtr> nk(k_.size()), nv(v_.size());
        std::vector<uint8_t*> nkp(k_.size()), nvp(v_.size());
        for (size_t l = 0; l < k_.size(); ++l) {
            nk[l] = owner_->alloc(kbytes);
            nv[l] = owner_->alloc(vbytes);
            if (k_[l]) {
                owner_->copy(*nk[l], 0, *k_[l], 0, k_[l]->size());
                owner_->copy(*nv[l], 0, *v_[l], 0, v_[l]->size());
            }
            nkp[l] = host_bytes(*nk[l]);
            nvp[l] = host_bytes(*nv[l]);
        }
        peak_ = std::max(peak_, add(allocated_bytes(), held));
        k_.swap(nk);
        v_.swap(nv);
        kp_.swap(nkp);
        vp_.swap(nvp);
        backed_ = want;
    }

    // Block `id` of a layer as bytes, and as the element type it holds; the caller checks the type and casts once per block.
    uint8_t* kraw(size_t layer, int32_t id) { return kp_[layer] + (size_t)id * k_block_bytes(); }
    uint8_t* vraw(size_t layer, int32_t id) { return vp_[layer] + (size_t)id * v_block_bytes(); }
    const uint8_t* kraw(size_t layer, int32_t id) const { return kp_[layer] + (size_t)id * k_block_bytes(); }
    const uint8_t* vraw(size_t layer, int32_t id) const { return vp_[layer] + (size_t)id * v_block_bytes(); }
    float* k(size_t layer, int32_t id) { return (float*)kraw(layer, id); }
    float* v(size_t layer, int32_t id) { return (float*)vraw(layer, id); }
    const float* k(size_t layer, int32_t id) const { return (const float*)kraw(layer, id); }
    const float* v(size_t layer, int32_t id) const { return (const float*)vraw(layer, id); }
    uint16_t* kh(size_t layer, int32_t id) { return (uint16_t*)kraw(layer, id); }
    uint16_t* vh(size_t layer, int32_t id) { return (uint16_t*)vraw(layer, id); }
    const uint16_t* kh(size_t layer, int32_t id) const { return (const uint16_t*)kraw(layer, id); }
    const uint16_t* vh(size_t layer, int32_t id) const { return (const uint16_t*)vraw(layer, id); }

private:
    // Resolved once per growth rather than per access: attention walks the block table for every head of every query.
    static uint8_t* host_bytes(Buffer& b) {
        auto* cpu = dynamic_cast<CpuBuffer*>(&b);
        if (!cpu) throw std::runtime_error("backend: KV storage needs host blocks");
        return (uint8_t*)cpu->host_address();
    }

    Backend* owner_;
    size_t heads_, dim_, max_, backed_ = 0, peak_ = 0;
    KVType kt_, vt_;
    size_t kb_, vb_;
    std::vector<BufferPtr> k_, v_;
    std::vector<uint8_t*> kp_, vp_;
};

// CPU implementation of the Backend interface.
// Uses AVX2 fused dequant+FMA for quantized matmuls where the host supports it, otherwise a scalar fallback.
class CpuBackend : public Backend {
public:
    CpuBackend() {
        unsigned hw = std::thread::hardware_concurrency();
        threads_ = (hw > 0) ? (int)hw : 4;
        if (threads_ > 64) threads_ = 64;
        avx2_ = has_avx2();   // detect once, not per row dot
        f16c_ = has_f16c();
        start_pool();
    }

    ~CpuBackend() override { stop_pool(); }

    void set_threads(int n) override {
        int t = (n > 0) ? n : 1;
        if (t == threads_) return;
        if (prefill_active_) throw std::runtime_error("CPU threads cannot change during prefill");
        stop_pool();
        threads_ = t;
        start_pool();
    }

    int threads_available() const override { return threads_; }

    // Weights on the host read the mapped file in place, so what counts against this is caches, activations and whatever a loader materializes.
    size_t memory_available() const override { return core::host_memory_available(); }

    void run_prefill(const std::function<void()>& work) override {
        if (prefill_active_) throw std::runtime_error("Nested CPU prefill is unsupported");
        prefill_active_ = true;
        try {
            detail::PrefillPlacement placement(threads_ == 6);
            auto finish = [&](std::exception_ptr error) {
                if (placement.needs_restore()) {
                    try { run_parallel([&](int w) { placement.restore(w); }); }
                    catch (...) {
                        // A retry may repair scheduling, but cannot erase the checked failure.
                        auto cleanup_error = std::current_exception();
                        try { run_parallel([&](int w) { placement.restore(w); }); }
                        catch (...) {}
                        std::rethrow_exception(cleanup_error);
                    }
                }
                if (error) std::rethrow_exception(error);
            };
            try {
                if (placement.enabled())
                    run_parallel([&](int w) { placement.apply(w); });
            } catch (...) { finish(std::current_exception()); }
            if (placement.enabled() && !placement.applied()) finish(nullptr);
            try { work(); }
            catch (...) { finish(std::current_exception()); }
            finish(nullptr);
            prefill_active_ = false;
        } catch (...) {
            prefill_active_ = false;
            throw;
        }
    }

    // The Q8_0 single-column path, kept as a private detail of this backend now that the interface is type-generic.
    // A scalar return per row was one kernel launch per row for a device backend, which is why it left the interface; it is still the right shape for the host.
    void matvec_q8_0(const uint8_t* data, const float* x, float* out,
                     size_t nblocks, size_t nout) {
        const int nt = threads_;
        // Small problems are not worth waking the pool.
        if (nt <= 1 || nout < (size_t)nt * 8) {
            for (size_t o = 0; o < nout; o++)
                out[o] = dot_row_impl(data + o * nblocks * gguf::Q8_0_TYPESIZE, x, nblocks);
            return;
        }
        const size_t chunk = (nout + (size_t)nt - 1) / (size_t)nt;
        run_parallel([&](int w) {
            size_t start = (size_t)w * chunk;
            size_t end = std::min(nout, start + chunk);
            for (size_t o = start; o < end; o++)
                out[o] = dot_row_impl(data + o * nblocks * gguf::Q8_0_TYPESIZE, x, nblocks);
        });
    }

    // Run fn(0..threads_-1) across the pool: worker 0 is the calling thread, so a single-threaded backend never touches the pool at all.
    // Blocks until every participant has returned, which is what lets the job be referenced rather than copied.
    template <class F>
    void run_parallel(F&& fn) {
        if (threads_ <= 1) { fn(0); return; }
        std::function<void(int)> job(std::ref(fn));   // ref -> no heap alloc
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &job;
            pending_.store(threads_ - 1, std::memory_order_relaxed);
            epoch_++;
        }
        cv_work_.notify_all();
        std::exception_ptr error;
        try { fn(0); }
        catch (...) { error = std::current_exception(); }
        for (int i = 0; i < SPIN_LIMIT && pending_.load(std::memory_order_acquire); i++)
            _mm_pause();
        std::unique_lock<std::mutex> lk(m_);
        // Acquire, not relaxed: only the last worker takes the mutex to notify, so the waiter must acquire on pending_ itself to see a non-last worker's results. This is the path after the spin budget ran out.
        cv_done_.wait(lk, [&] { return pending_.load(std::memory_order_acquire) == 0; });
        job_ = nullptr;
        if (!error) error = worker_error_;
        worker_error_ = nullptr;
        lk.unlock();
        // The borrowed callable must stay alive until all workers finish, including when the calling participant fails.
        if (error) std::rethrow_exception(error);
    }

    void embed(Slice dst_s, uint32_t type, CSlice table, size_t nin,
               size_t nrows, const uint32_t* ids, size_t count) override {
        float* dst = at(dst_s);
        const uint8_t* rows = (const uint8_t*)bytes_at(table);
        const quant::QuantType* qt = type == gguf::GGML_TYPE_F32
                                   ? nullptr : quant::Registry::instance().get(type);
        if (type != gguf::GGML_TYPE_F32 && (!qt || !qt->dequantize))
            throw std::runtime_error("backend: unsupported embedding type");
        const size_t stride = qt ? (nin / qt->block_size) * qt->type_size : nin * sizeof(float);
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] >= nrows) throw std::runtime_error("backend: embedding row out of range");
            const uint8_t* row = rows + (size_t)ids[i] * stride;
            if (qt) qt->dequantize(row, dst + i * nin, nin / qt->block_size);
            else std::memcpy(dst + i * nin, row, nin * sizeof(float));
        }
    }

    // Host memory is host visible whatever was asked for.
    BufferPtr alloc(size_t bytes, Memory) override { return std::make_shared<CpuBuffer>(bytes); }

    BufferPtr adopt(const void* src, size_t bytes) override {
        if (!src && bytes) throw std::runtime_error("backend: adopting null storage");
        return std::make_shared<CpuBuffer>(src, bytes);
    }

    // Eager: an op has completed by the time it returns, so there is never anything outstanding to wait for, and a ticket only counts.
    Ticket submit() override { return ++ticket_; }
    void wait(Ticket) noexcept override {}
    void sync() noexcept override {}

    void read(const Buffer& src, size_t off, void* dst, size_t bytes) override {
        span(src, off, bytes);
        std::memcpy(dst, (const uint8_t*)src.host_ptr() + off, bytes);
    }

    void write(Buffer& dst, size_t off, const void* src, size_t bytes) override {
        if (!src && bytes) throw std::runtime_error("backend: writing from null storage");
        span(dst, off, bytes);
        std::memcpy((uint8_t*)host(dst) + off, src, bytes);
    }

    void copy(Buffer& dst, size_t dst_off, const Buffer& src, size_t src_off,
              size_t bytes) override {
        span(dst, dst_off, bytes);
        span(src, src_off, bytes);
        std::memcpy((uint8_t*)host(dst) + dst_off,
                    (const uint8_t*)src.host_ptr() + src_off, bytes);
    }

    // The product lands in a scratch buffer kept across calls and is added to Y, so the arithmetic is the separate matmul and add exactly.
    void matmul_add(uint32_t type, CSlice weights, CSlice X_s, Slice Y_s,
                    size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}) override {
        const size_t bytes = nout * nbatch * sizeof(float);
        if (!add_scratch_ || add_scratch_bytes_ < bytes) {
            add_scratch_ = alloc(bytes, Memory::device);
            add_scratch_bytes_ = bytes;
        }
        matmul(type, weights, X_s, {add_scratch_.get(), 0}, nin, nout, nbatch, runs);
        add(Y_s, {add_scratch_.get(), 0}, nout * nbatch);
    }
    BufferPtr add_scratch_;
    size_t add_scratch_bytes_ = 0;

    // With row runs a generated token (extent 1) takes the decode dots and a prompt's rows the batched path, so a row computes the same however it is batched, as on a device (backend.hpp RowRuns); without them a one-column call is decode.
    void matmul(uint32_t type, CSlice weights, CSlice X_s, Slice Y_s,
                size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}) override {
        const uint8_t* data = (const uint8_t*)bytes_at(weights);
        const float* X = at(X_s);
        float* Y = at(Y_s);
        each_run(nbatch, runs, [&](size_t first, size_t count, bool decode) {
            matmul_raw(type, data, X + first * nin, Y + first * nout, nin, nout, count, decode);
        });
    }

    // Calls `each(first, count, decode)` over stretches of rows that take the same path.
    template <typename Fn>
    static void each_run(size_t nbatch, RowRuns runs, const Fn& each) {
        if (!nbatch) return;
        if (!runs.n) { each(0, nbatch, nbatch == 1); return; }
        if (runs.runs[runs.n - 1].end != nbatch) throw std::runtime_error("backend: row runs do not cover the batch");
        size_t start = 0;
        for (size_t i = 0; i < runs.n;) {
            const bool decode = runs.runs[i].extent <= 1;
            size_t j = i + 1;
            while (j < runs.n && (runs.runs[j].extent <= 1) == decode) ++j;
            const size_t end = runs.runs[j - 1].end;
            if (end < start) throw std::runtime_error("backend: row runs out of order");
            if (end > start) each(start, end - start, decode);
            start = end;
            i = j;
        }
    }

    // Whether decode rows meet quantized activations (q8_dots.hpp); a reference that wants the float arithmetic turns it off.
    void set_decode_activations8(bool on) { decode8_ = on; }

    // Decode columns of X against a quantized matrix through the 8-bit dots: every column quantized once, the rows of all of them split over the pool.
    void matvec_q8x(uint32_t type, const uint8_t* data, const float* X, float* Y, size_t nin, size_t nout, size_t ncols) {
        xq8_.reset(X, ncols, nin);
        prepare_x(type);
        const size_t rb = row_bytes_of(type, nin), rows = ncols * nout;
        auto work = [&](size_t r0, size_t r1) {
            for (size_t r = r0; r < r1; ++r) {
                const size_t c = r / nout, o = r - c * nout;
                Y[r] = q8::dot(type, data + o * rb, xq8_, c);
            }
        };
        const size_t nt = (size_t)std::max(threads_, 1);
        if (nt <= 1 || rows < nt * 8) { work(0, rows); return; }
        const size_t chunk = (rows + nt - 1) / nt;
        run_parallel([&](int w) { work(std::min(rows, (size_t)w * chunk), std::min(rows, (size_t)(w + 1) * chunk)); });
    }

    static constexpr size_t kPromptDotsFrom = 4096;   // the narrowest K-quant row a prompt meets through the prompt dots (matmul_raw)

    // A prompt's columns of X against a quantized matrix through the prompt dots (q8_dots.hpp dot_block): every column quantized once, stretches of 16 rows handed to workers as they free up, each against every column, so a row's weights are unpacked once for all of them.
    void matmul_q8_prompt(uint32_t type, const uint8_t* data, const float* X, float* Y, size_t nin, size_t nout, size_t ncols) {
        xq8_.reset(X, ncols, nin);
        prepare_x(type);
        const size_t rb = row_bytes_of(type, nin), R = 16, tasks = (nout + R - 1) / R;
        prompt_rows_.resize(ncols);
        prompt_outs_.resize(ncols);
        for (size_t c = 0; c < ncols; ++c) {
            prompt_rows_[c] = c;
            prompt_outs_[c] = Y + c * nout;
        }
        std::atomic<size_t> next{0};
        auto work = [&](int) {
            for (size_t t = next.fetch_add(1); t < tasks; t = next.fetch_add(1)) {
                const size_t o0 = t * R, o1 = std::min(nout, o0 + R);
                q8::dot_block(type, data + o0 * rb, rb, o1 - o0, xq8_, prompt_rows_.data(), ncols, prompt_outs_.data(), o0);
            }
        };
        if (threads_ <= 1 || tasks < 2) work(0);
        else run_parallel(work);
    }

    // The matmul on host addresses, which an expert's matrix inside a stacked tensor needs; `decode` picks the decode dots, and otherwise a type with quantized dots takes the prompt dots and the rest the batched float path.
    void matmul_raw(uint32_t type, const uint8_t* data, const float* X, float* Y,
                    size_t nin, size_t nout, size_t nbatch, bool decode) {
        if (decode && quantized_dots(type, nin)) {
            matvec_q8x(type, data, X, Y, nin, nout, nbatch);
            return;
        }
        // The K-quants' float path dequantizes every row block before its dots, and on rows at least kPromptDotsFrom wide it loses to the prompt dots; narrower rows' dequantized blocks stay in the first-level cache, where the float path's 4-row by 3-column register blocking kept Qwen3-0.6B ahead. Q8_0, Q4_0 and Q4_1 dequantize cheaply and keep the float path (docs/src/backends-cpu.md).
        // The choice follows the matrix alone, so a prompt's row computes the same however it is batched.
        if (!decode && quantized_dots(type, nin) && is_kquant(type) && nin >= kPromptDotsFrom) {
            matmul_q8_prompt(type, data, X, Y, nin, nout, nbatch);
            return;
        }
        if (decode && nbatch > 1) {
            for (size_t c = 0; c < nbatch; ++c) matmul_raw(type, data, X + c * nin, Y + c * nout, nin, nout, 1, true);
            return;
        }
        // Q8_0 keeps its fused dequant+FMA row dot for the single-column case, which is the decode path and is bandwidth bound rather than load bound, so the extra dequant buffer would buy nothing there.
        if (decode && type == gguf::GGML_TYPE_Q8_0) {
            matvec_q8_0(data, X, Y, nin / gguf::Q8_0_BLOCK, nout);
            return;
        }
        // K-quants whose dot factorises so no dequantized value is materialised: Q4_K/Q5_K give d*sum(q*x) - m*sum(x), Q6_K has signed group scales and no min, so it is sum over groups of d_g*sum(q*x).
        if (decode && (type == gguf::GGML_TYPE_Q4_K ||
                            type == gguf::GGML_TYPE_Q5_K ||
                            type == gguf::GGML_TYPE_Q6_K)) {
            const size_t blk = type == gguf::GGML_TYPE_Q4_K ? gguf::Q4_K_BLOCK
                             : type == gguf::GGML_TYPE_Q5_K ? gguf::Q5_K_BLOCK
                                                                 : gguf::Q6_K_BLOCK;
            const size_t tsz = type == gguf::GGML_TYPE_Q4_K ? gguf::Q4_K_TYPESIZE
                             : type == gguf::GGML_TYPE_Q5_K ? gguf::Q5_K_TYPESIZE
                                                                 : gguf::Q6_K_TYPESIZE;
            const size_t nb = nin / blk;
            const size_t rowbytes = nb * tsz;
            // A fused dot applies the block scale after sum(q*x), so a large activation can overflow the inner sum where dequantizing first stays finite (then d*Inf is Inf, and 0*Inf NaN).
            // Once any partial overflows the row result is Inf or NaN, never a plausible finite number, so a finite fused result needs no fallback.
            const auto dot = [&](const uint8_t* r) {
                float v;
                switch (type) {
                    case gguf::GGML_TYPE_Q4_K: v = dot_row_q4_K(r, X, nb); break;
                    case gguf::GGML_TYPE_Q5_K: v = dot_row_q5_K(r, X, nb); break;
                    default:                   v = dot_row_q6_K(r, X, nb); break;
                }
                return std::isfinite(v) ? v : dot_row_dequant(type, r, X, nin, nb);
            };
            const int nt = threads_;
            if (nt <= 1 || nout < (size_t)nt * 8) {
                for (size_t o = 0; o < nout; o++)
                    Y[o] = dot(data + o * rowbytes);
                return;
            }
            const size_t chunk = (nout + (size_t)nt - 1) / (size_t)nt;
            run_parallel([&](int w) {
                const size_t s0 = (size_t)w * chunk;
                const size_t e0 = std::min(nout, s0 + chunk);
                for (size_t o = s0; o < e0; o++)
                    Y[o] = dot(data + o * rowbytes);
            });
            return;
        }
        const quant::QuantType* qt = quant::Registry::instance().get(type);
        const bool f32 = type == gguf::GGML_TYPE_F32;
        if (!f32 && (!qt || !qt->dequantize || qt->block_size == 0))
            throw std::runtime_error("backend: no dequantizer for tensor type");
        const size_t nblocks = f32 ? 0 : nin / qt->block_size;
        const size_t rowbytes = f32 ? nin * sizeof(float) : nblocks * qt->type_size;

        // Rows dequantized together before walking the batch: the fused kernel's row width, since dot_f32_x4 shares one activation load across exactly 4 rows.
        // A cache-byte budget measured worse at every size, because the knee follows the kernel width, not the working set.
        const size_t RB = (size_t)DOT_ROWS;

        auto do_rows = [&](int w, size_t o0, size_t o1) {
            if (f32 && decode) {
                // Decode streams each resident row contiguously; prefill keeps the fused kernels that reuse weights across batch columns.
                for (size_t o = o0; o < o1; ++o)
                    Y[o] = dot_f32((const float*)(data + o * rowbytes), X, nin);
                return;
            }
            std::vector<float>& buf = rowbuf_[(size_t)w];
            if (!f32 && buf.size() < RB * nin) buf.assign(RB * nin, 0.0f);
            for (size_t o = o0; o < o1; o += RB) {
                const size_t nr = std::min(RB, o1 - o);
                const float* r = f32 ? (const float*)(data + o * rowbytes) : buf.data();
                if (!f32)
                    for (size_t k = 0; k < nr; k++)
                        qt->dequantize(data + (o + k) * rowbytes, buf.data() + k * nin, nblocks);
                size_t b = 0;
                // Three activation columns at a time where the row block is full, so weight loads amortise across all three.
                for (; b + 3 <= nbatch && nr == RB; b += 3) {
                    const float* xa = X + b * nin;
                    const float* xb3 = X + (b + 1) * nin;
                    const float* xc3 = X + (b + 2) * nin;
                    float* ya = Y + b * nout + o;
                    float* yb = Y + (b + 1) * nout + o;
                    float* yc = Y + (b + 2) * nout + o;
                    for (size_t k = 0; k + 4 <= nr; k += 4)
                        dot_f32_x4x3(r + k * nin, nin, xa, xb3, xc3, nin,
                                     ya + k, yb + k, yc + k);
                }
                for (; b + 2 <= nbatch && nr == RB; b += 2) {
                    const float* xa = X + b * nin;
                    const float* xb2 = X + (b + 1) * nin;
                    float* ya = Y + b * nout + o;
                    float* yb = Y + (b + 1) * nout + o;
                    for (size_t k = 0; k + 4 <= nr; k += 4)
                        dot_f32_x4x2(r + k * nin, nin, xa, xb2, nin, ya + k, yb + k);
                }
                for (; b < nbatch; b++) {
                    const float* x = X + b * nin;
                    float* y = Y + b * nout + o;
                    size_t k = 0;
                    for (; k + 4 <= nr; k += 4)
                        dot_f32_x4(r + k * nin, nin, x, nin, y + k);
                    for (; k < nr; k++)
                        y[k] = dot_f32(r + k * nin, x, nin);
                }
            }
        };
        const int nt = threads_;
        if (nt <= 1 || nout < (size_t)nt * 4) { do_rows(0, 0, nout); return; }
        const size_t per = (nout + (size_t)nt - 1) / (size_t)nt;
        const size_t chunk = (per + RB - 1) / RB * RB;
        run_parallel([&](int w) {
            const size_t s = (size_t)w * chunk;
            const size_t e = std::min(nout, s + chunk);
            if (s < e) do_rows(w, s, e);
        });
    }

    void matmul_group(std::initializer_list<Projection> projections,
                      CSlice X_s, size_t nin, size_t nbatch, RowRuns runs = {}) override {
        for (const auto& p : projections)
            if (!p.data.buffer) throw std::runtime_error("backend: projection without storage");
        bool decode = runs.n ? true : nbatch == 1;
        for (size_t i = 0; i < runs.n; ++i) decode = decode && runs.runs[i].extent <= 1;
        bool q8 = decode && decode8_ && avx2_ && nin % 32 == 0 && projections.size() > 1;
        for (const auto& p : projections) q8 = q8 && q8::has_dot(p.type);
        if (!q8) { Backend::matmul_group(projections, X_s, nin, nbatch, runs); return; }
        // One quantized X for every projection, and one pool dispatch over all their rows.
        xq8_.reset(at(X_s), nbatch, nin);
        for (const auto& p : projections) prepare_x(p.type);
        struct Part { uint32_t type; const uint8_t* data; float* out; size_t rows, row_bytes, first; };
        std::vector<Part> parts;
        size_t total = 0;
        for (const auto& p : projections) {
            parts.push_back({p.type, (const uint8_t*)bytes_at(p.data), at(p.out), p.rows, row_bytes_of(p.type, nin), total});
            total += nbatch * p.rows;
        }
        auto work = [&](size_t r0, size_t r1) {
            for (const Part& p : parts) {
                const size_t lo = std::max(r0, p.first), hi = std::min(r1, p.first + nbatch * p.rows);
                for (size_t r = lo; r < hi; ++r) {
                    const size_t c = (r - p.first) / p.rows, o = (r - p.first) - c * p.rows;
                    p.out[c * p.rows + o] = q8::dot(p.type, p.data + o * p.row_bytes, xq8_, c);
                }
            }
        };
        const size_t nt = (size_t)std::max(threads_, 1);
        if (nt <= 1 || total < nt * 8) { work(0, total); return; }
        const size_t chunk = (total + nt - 1) / nt;
        run_parallel([&](int w) { work(std::min(total, (size_t)w * chunk), std::min(total, (size_t)(w + 1) * chunk)); });
    }

    // Rows fused per activation load: the width dot_f32_x4 handles.
    static const int DOT_ROWS = 4;


    // Four rows against THREE activation columns: 7 loads per 12 FMAs, a ratio of 0.58 against 0.75 for the two-column form.
    // This needs 12 accumulators plus 3 activation registers, which is exactly the 16 YMM budget, so it sits on the spill boundary and is only worth keeping if measured faster.
    static void dot_f32_x4x3(const float* r, size_t stride,
                             const float* xa, const float* xb, const float* xc,
                             size_t n, float* outa, float* outb, float* outc) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
        const float* r0 = r;
        const float* r1 = r + stride;
        const float* r2 = r + 2 * stride;
        const float* r3 = r + 3 * stride;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(xa + i);
            const __m256 yv = _mm256_loadu_ps(xb + i);
            const __m256 zv = _mm256_loadu_ps(xc + i);
            __m256 w = _mm256_loadu_ps(r0 + i);
            a0 = _mm256_fmadd_ps(w, xv, a0); b0 = _mm256_fmadd_ps(w, yv, b0); c0 = _mm256_fmadd_ps(w, zv, c0);
            w = _mm256_loadu_ps(r1 + i);
            a1 = _mm256_fmadd_ps(w, xv, a1); b1 = _mm256_fmadd_ps(w, yv, b1); c1 = _mm256_fmadd_ps(w, zv, c1);
            w = _mm256_loadu_ps(r2 + i);
            a2 = _mm256_fmadd_ps(w, xv, a2); b2 = _mm256_fmadd_ps(w, yv, b2); c2 = _mm256_fmadd_ps(w, zv, c2);
            w = _mm256_loadu_ps(r3 + i);
            a3 = _mm256_fmadd_ps(w, xv, a3); b3 = _mm256_fmadd_ps(w, yv, b3); c3 = _mm256_fmadd_ps(w, zv, c3);
        }
        // Keep lane order without making every accumulator addressable on the stack.
        const auto finish = [&](const __m256 acc, const float* row, const float* x) {
            const __m128 lo = _mm256_castps256_ps128(acc);
            const __m128 hi = _mm256_extractf128_ps(acc, 1);
            float v = _mm_cvtss_f32(lo);
            v += _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 1, 1, 1)));
            v += _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 2, 2, 2)));
            v += _mm_cvtss_f32(_mm_shuffle_ps(lo, lo, _MM_SHUFFLE(3, 3, 3, 3)));
            v += _mm_cvtss_f32(hi);
            v += _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, _MM_SHUFFLE(1, 1, 1, 1)));
            v += _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2, 2, 2, 2)));
            v += _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, _MM_SHUFFLE(3, 3, 3, 3)));
            for (size_t j = i; j < n; ++j) v += row[j] * x[j];
            return v;
        };
        outa[0] = finish(a0, r0, xa);
        outa[1] = finish(a1, r1, xa);
        outa[2] = finish(a2, r2, xa);
        outa[3] = finish(a3, r3, xa);
        outb[0] = finish(b0, r0, xb);
        outb[1] = finish(b1, r1, xb);
        outb[2] = finish(b2, r2, xb);
        outb[3] = finish(b3, r3, xb);
        outc[0] = finish(c0, r0, xc);
        outc[1] = finish(c1, r1, xc);
        outc[2] = finish(c2, r2, xc);
        outc[3] = finish(c3, r3, xc);
    }

    // Four rows against TWO activation columns in one pass. dot_f32_x4 costs 5 loads per 4 FMAs (one activation, four weights).
    // Holding two activation columns makes it 6 loads per 8 FMAs, so the load:FMA ratio drops from 1.25 to 0.75 and the kernel stops being load bound.
    // Eight accumulators plus two activation registers still fit the 16 YMM registers, so nothing spills.
    static void dot_f32_x4x2(const float* r, size_t stride,
                             const float* xa, const float* xb, size_t n,
                             float* outa, float* outb) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        const float* r0 = r;
        const float* r1 = r + stride;
        const float* r2 = r + 2 * stride;
        const float* r3 = r + 3 * stride;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(xa + i);
            const __m256 yv = _mm256_loadu_ps(xb + i);
            __m256 w = _mm256_loadu_ps(r0 + i);
            a0 = _mm256_fmadd_ps(w, xv, a0); b0 = _mm256_fmadd_ps(w, yv, b0);
            w = _mm256_loadu_ps(r1 + i);
            a1 = _mm256_fmadd_ps(w, xv, a1); b1 = _mm256_fmadd_ps(w, yv, b1);
            w = _mm256_loadu_ps(r2 + i);
            a2 = _mm256_fmadd_ps(w, xv, a2); b2 = _mm256_fmadd_ps(w, yv, b2);
            w = _mm256_loadu_ps(r3 + i);
            a3 = _mm256_fmadd_ps(w, xv, a3); b3 = _mm256_fmadd_ps(w, yv, b3);
        }
        alignas(32) float t[8];
        const __m256* accs[8] = { &a0, &a1, &a2, &a3, &b0, &b1, &b2, &b3 };
        for (int k = 0; k < 8; k++) {
            _mm256_store_ps(t, *accs[k]);
            float v = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            const size_t row = (size_t)(k & 3);
            const float* xt = (k < 4) ? xa : xb;
            for (size_t j = i; j < n; j++) v += r[row * stride + j] * xt[j];
            ((k < 4) ? outa : outb)[row] = v;
        }
    }

    // Four dots against a SHARED activation vector, in one pass.
    // Calling dot_f32 four times costs 2 loads per FMA (one weight, one activation), and Zen3 sustains 2 loads/cycle against 2 FMAs/cycle, so that kernel is load bound at half of FMA peak.
    // Loading x once and reusing it across 4 rows costs 5 loads per 4 FMAs instead of 8.
    static void dot_f32_x4(const float* r, size_t stride, const float* x,
                           size_t n, float* out) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        const float* r0 = r;
        const float* r1 = r + stride;
        const float* r2 = r + 2 * stride;
        const float* r3 = r + 3 * stride;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(x + i);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), xv, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), xv, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), xv, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), xv, s3);
        }
        alignas(32) float t[8];
        const __m256* acc[4] = { &s0, &s1, &s2, &s3 };
        for (int k = 0; k < 4; k++) {
            _mm256_store_ps(t, *acc[k]);
            float v = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            for (size_t j = i; j < n; j++) v += r[(size_t)k * stride + j] * x[j];
            out[k] = v;
        }
    }

    // Four independent accumulators.
    // With a single accumulator every FMA depends on the previous one, so the loop runs at FMA LATENCY (about 4 cycles) instead of FMA throughput (about 0.5), which is most of an order of magnitude on this path.
    // Splitting the chain also changes the summation order, so results differ in the last bits.
    static float dot_f32(const float* a, const float* b, size_t n) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8), s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        for (; i + 8 <= n; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float out = _mm_cvtss_f32(s);
        for (; i < n; i++) out += a[i] * b[i];
        return out;
    }

    // CPU-only: a host callback across host threads has no device analogue, so this is not on the Backend interface.
    // The batched ops above are how the model gets parallelism; this stays public for the backend's own tests.
    void parallel_for(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        const int nt = std::min(threads_, n);
        if (nt <= 1) { for (int i = 0; i < n; i++) fn(i); return; }
        const int chunk = (n + nt - 1) / nt;
        run_parallel([&](int w) {
            const int start = w * chunk;
            const int end = std::min(n, start + chunk);
            for (int i = start; i < end; i++) fn(i);
        });
    }

    KVLayout kv_layout() const override { return {KV_BLOCK_TOKENS}; }

    std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim,
                                        size_t max_tokens, KVType k_type = KVType::f32,
                                        KVType v_type = KVType::f32) override {
        if (layers == 0 || n_head_kv == 0 || head_dim == 0)
            throw std::runtime_error("backend: invalid KV storage shape");
        const size_t blocks = CpuKVStorage::blocks_for(max_tokens);
        // The whole budget must be addressable, even if it is never backed;
        // every factor goes through the checked multiply.
        using S = CpuKVStorage;
        S::mul(S::mul(S::mul(S::mul(blocks, layers), 2), n_head_kv),
               S::mul(S::mul(KV_BLOCK_TOKENS, head_dim), sizeof(float)));
        return std::make_unique<CpuKVStorage>(*this, layers, n_head_kv, head_dim, blocks, k_type, v_type);
    }

    void kv_copy(KVStorage& storage, int32_t src, int32_t dst) override {
        auto* s = dynamic_cast<CpuKVStorage*>(&storage);
        if (!s) throw std::runtime_error("backend: KV storage of another backend");
        if (src < 0 || dst < 0 || !s->backed((size_t)src) || (size_t)dst >= s->max_blocks())
            throw std::runtime_error("backend: KV copy outside the storage");
        s->ensure((size_t)dst);
        for (size_t l = 0; l < s->layers(); ++l) {
            std::copy_n(s->kraw(l, src), s->k_block_bytes(), s->kraw(l, dst));
            std::copy_n(s->vraw(l, src), s->v_block_bytes(), s->vraw(l, dst));
        }
    }

    // A row of floats into a cache side of either type; f16 rounds to nearest, eight at a time where F16C is present.
    void kv_store(uint8_t* dst, KVType type, const float* src, size_t n) const {
        if (type == KVType::f32) { std::copy_n(src, n, (float*)dst); return; }
        uint16_t* d = (uint16_t*)dst;
        size_t i = 0;
        if (avx2_)
            for (; i + 8 <= n; i += 8)
                _mm_storeu_si128((__m128i*)(d + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
        for (; i < n; ++i) d[i] = f32_to_f16(src[i]);
    }
    // dot(q, k) over an f16 key row.
    float dot_f32_f16(const float* q, const uint16_t* k, size_t n) const {
        float sum = 0.0f;
        size_t i = 0;
        if (avx2_) {
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= n; i += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i),
                                      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(k + i))), acc);
            float lanes[8];
            _mm256_storeu_ps(lanes, acc);
            for (float x : lanes) sum += x;
        }
        for (; i < n; ++i) sum += q[i] * f16_to_f32(k[i]);
        return sum;
    }

    void kv_write(size_t layer, const KVView* views, size_t n_views,
                  CSlice k_s, CSlice v_s) override {
        if (n_views && !views) throw std::runtime_error("backend: KV write without views");
        const float* k = at(k_s);
        const float* v = at(v_s);
        size_t row0 = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            CpuKVStorage& s = storage_of(view);
            const size_t bt = KV_BLOCK_TOKENS, heads = s.heads(), dim = s.dim();
            const size_t pos = view.length, batch = view.nq;
            if (layer >= s.layers() ||
                CpuKVStorage::blocks_for(CpuKVStorage::add(pos, batch)) > view.n_blocks)
                throw std::runtime_error("backend: KV write outside the view");
            for (size_t b = 0; b < batch; ++b) {
                const size_t t = pos + b;
                const int32_t id = view.blocks[t / bt];
                s.ensure((size_t)id);
                for (size_t h = 0; h < heads; ++h) {
                    const size_t in = ((row0 + b) * heads + h) * dim;
                    const size_t out = (h * bt + t % bt) * dim;
                    kv_store(s.kraw(layer, id) + out * kv_elem_bytes(s.k_type()), s.k_type(), k + in, dim);
                    kv_store(s.vraw(layer, id) + out * kv_elem_bytes(s.v_type()), s.v_type(), v + in, dim);
                }
            }
            row0 += batch;
        }
    }

    // Blocks are walked in table order and every reduction keeps token order: one global softmax over the scores and per-lane value accumulation across block edges, so the arithmetic is that of a contiguous history.
    // Views are taken one after another: the per-view work is what it was for one sequence, so a single view computes exactly what it did before the batch form existed.
    void attention(CSlice Q_s, size_t layer, const KVView* views, size_t n_views,
                   Slice out_s, int n_head, int n_head_kv, int head_dim) override {
        if (n_views && !views) throw std::runtime_error("backend: attention without views");
        const float* Q_all = at(Q_s);
        float* out_all = at(out_s);
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0)
            throw std::runtime_error("backend: invalid attention dimensions");
        const size_t q_stride = (size_t)n_head * head_dim;
        const size_t hd = (size_t)head_dim;
        const int ratio = n_head / n_head_kv;
        const float scale = 1.0f / std::sqrt((float)head_dim);
        size_t row0 = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            if (!view.nq || view.nq > (size_t)std::numeric_limits<int>::max())
                throw std::runtime_error("backend: invalid attention dimensions");
            const int nbatch = (int)view.nq;
            const float* Q = Q_all + row0 * q_stride;
            float* out = out_all + row0 * q_stride;
            const CpuKVStorage& s = storage_of(view);
            const size_t bt = KV_BLOCK_TOKENS;
            const size_t sequence = CpuKVStorage::add(view.length, view.nq);
            const size_t blocks = CpuKVStorage::blocks_for(sequence);
            if (layer >= s.layers() || (size_t)head_dim != s.dim() ||
                (size_t)n_head_kv != s.heads() || blocks > view.n_blocks)
                throw std::runtime_error("backend: attention outside the KV view");
            for (size_t i = 0; i < blocks; ++i)
                if (!s.backed((size_t)view.blocks[i]))
                    throw std::runtime_error("backend: attention over unwritten KV blocks");
            attention_scores_.resize((size_t)n_head * sequence);
            parallel_for(n_head, [&](int h) {
                const size_t kvh = (size_t)(h / ratio);
                float* scores = attention_scores_.data() + (size_t)h * sequence;
                for (int b = 0; b < nbatch; ++b) {
                    const size_t end = view.length + (size_t)b + 1;
                    const float* q = Q + (size_t)b * q_stride + (size_t)h * hd;
                    float max_score = -std::numeric_limits<float>::infinity();
                    for (size_t t0 = 0; t0 < end; t0 += bt) {
                        const size_t n = std::min(bt, end - t0);
                        if (s.k_type() == KVType::f16) {
                            const uint16_t* kb = s.kh(layer, view.blocks[t0 / bt]) + kvh * bt * hd;
                            for (size_t j = 0; j < n; ++j) {
                                scores[t0 + j] = dot_f32_f16(q, kb + j * hd, hd) * scale;
                                max_score = std::max(max_score, scores[t0 + j]);
                            }
                            continue;
                        }
                        const float* kb = s.k(layer, view.blocks[t0 / bt]) + kvh * bt * hd;
                        for (size_t j = 0; j < n; ++j) {
                            const float* k = kb + j * hd;
                            float score = 0.0f;
                            if (avx2_) score = dot_f32(q, k, hd);
                            else for (size_t d = 0; d < hd; ++d) score += q[d] * k[d];
                            scores[t0 + j] = score * scale;
                            max_score = std::max(max_score, scores[t0 + j]);
                        }
                    }
                    float sum = 0.0f;
                    for (size_t t = 0; t < end; ++t) {
                        scores[t] = std::exp(scores[t] - max_score);
                        sum += scores[t];
                    }
                    float* dst = out + (size_t)b * q_stride + (size_t)h * hd;
                    // Normalize once; each lane then keeps sequence order while its partial sum stays in a register across KV rows.
                    for (size_t t = 0; t < end; ++t) scores[t] /= sum;
                    const auto vblock = [&](size_t t0) {
                        return s.v(layer, view.blocks[t0 / bt]) + kvh * bt * hd;
                    };
                    if (s.v_type() == KVType::f16) {
                        // Eight halves widened per load where F16C is present.
                        const auto vblock16 = [&](size_t t0) {
                            return s.vh(layer, view.blocks[t0 / bt]) + kvh * bt * hd;
                        };
                        size_t d = 0;
                        if (avx2_) {
                            for (; d + 8 <= hd; d += 8) {
                                __m256 acc = _mm256_setzero_ps();
                                for (size_t t0 = 0; t0 < end; t0 += bt) {
                                    const uint16_t* vb = vblock16(t0) + d;
                                    const size_t n = std::min(bt, end - t0);
                                    for (size_t j = 0; j < n; ++j)
                                        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(scores[t0 + j]),
                                                                               _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(vb + j * hd)))));
                                }
                                _mm256_storeu_ps(dst + d, acc);
                            }
                        }
                        for (; d < hd; ++d) {
                            float acc = 0.0f;
                            for (size_t t0 = 0; t0 < end; t0 += bt) {
                                const uint16_t* vb = vblock16(t0) + d;
                                const size_t n = std::min(bt, end - t0);
                                for (size_t j = 0; j < n; ++j) acc += scores[t0 + j] * f16_to_f32(vb[j * hd]);
                            }
                            dst[d] = acc;
                        }
                        continue;
                    }
                    size_t d = 0;
                    if (avx2_) {
                        for (; d + 32 <= hd; d += 32) {
                            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                            for (size_t t0 = 0; t0 < end; t0 += bt) {
                                const float* vb = vblock(t0) + d;
                                const size_t n = std::min(bt, end - t0);
                                for (size_t j = 0; j < n; ++j) {
                                    const float* v = vb + j * hd;
                                    const __m256 sw = _mm256_set1_ps(scores[t0 + j]);
                                    a0 = _mm256_add_ps(a0, _mm256_mul_ps(sw, _mm256_loadu_ps(v)));
                                    a1 = _mm256_add_ps(a1, _mm256_mul_ps(sw, _mm256_loadu_ps(v + 8)));
                                    a2 = _mm256_add_ps(a2, _mm256_mul_ps(sw, _mm256_loadu_ps(v + 16)));
                                    a3 = _mm256_add_ps(a3, _mm256_mul_ps(sw, _mm256_loadu_ps(v + 24)));
                                }
                            }
                            _mm256_storeu_ps(dst + d, a0); _mm256_storeu_ps(dst + d + 8, a1);
                            _mm256_storeu_ps(dst + d + 16, a2); _mm256_storeu_ps(dst + d + 24, a3);
                        }
                        for (; d + 8 <= hd; d += 8) {
                            __m256 acc = _mm256_setzero_ps();
                            for (size_t t0 = 0; t0 < end; t0 += bt) {
                                const float* vb = vblock(t0) + d;
                                const size_t n = std::min(bt, end - t0);
                                for (size_t j = 0; j < n; ++j)
                                    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(scores[t0 + j]),
                                                                           _mm256_loadu_ps(vb + j * hd)));
                            }
                            _mm256_storeu_ps(dst + d, acc);
                        }
                    }
                    for (; d < hd; ++d) {
                        float acc = 0.0f;
                        for (size_t t0 = 0; t0 < end; t0 += bt) {
                            const float* vb = vblock(t0) + d;
                            const size_t n = std::min(bt, end - t0);
                            for (size_t j = 0; j < n; ++j) acc += scores[t0 + j] * vb[j * hd];
                        }
                        dst[d] = acc;
                    }
                }
            });
            row0 += view.nq;
        }
    }

    void rms_norm(Slice dst_s, CSlice src_s, CSlice w_s, size_t n, float eps) override {
        rms_norm_raw(at(dst_s), at(src_s), at(w_s), n, eps);
    }

    void rms_norm_raw(float* dst, const float* src, const float* w, size_t n, float eps) {
        if (avx2_) {
            // Sum of squares (vectorized), then a vectorized weighted scale.
            __m256 acc = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(src + i);
                acc = _mm256_fmadd_ps(x, x, acc);
            }
            __m256 h = _mm256_hadd_ps(acc, acc);
            h = _mm256_hadd_ps(h, h);
            float s = _mm256_cvtss_f32(h);
            s += _mm_cvtss_f32(_mm256_extractf128_ps(h, 1));
            for (; i < n; i++) s += src[i] * src[i];
            float r = 1.0f / std::sqrt(s / (float)n + eps);
            __m256 rv = _mm256_set1_ps(r);
            i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(src + i);
                __m256 wv = _mm256_loadu_ps(w + i);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, _mm256_mul_ps(rv, wv)));
            }
            for (; i < n; i++) dst[i] = src[i] * r * w[i];
            return;
        }
        float s = 0.0f;
        for (size_t i = 0; i < n; i++) s += src[i] * src[i];
        float r = 1.0f / std::sqrt(s / (float)n + eps);
        for (size_t i = 0; i < n; i++) dst[i] = src[i] * r * w[i];
    }

    void rope_raw(float* x, const float* cos, const float* sin, int half) const {
        if (avx2_) {
            int i = 0;
            for (; i + 8 <= half; i += 8) {
                __m256 xa = _mm256_loadu_ps(x + i);
                __m256 xb = _mm256_loadu_ps(x + i + half);
                __m256 c = _mm256_loadu_ps(cos + i);
                __m256 s = _mm256_loadu_ps(sin + i);
                __m256 na = _mm256_fnmadd_ps(xb, s, _mm256_mul_ps(xa, c));
                __m256 nb = _mm256_fmadd_ps(xa, s, _mm256_mul_ps(xb, c));
                _mm256_storeu_ps(x + i, na);
                _mm256_storeu_ps(x + i + half, nb);
            }
            for (; i < half; i++) {
                int a = i, b = i + half;
                float xa = x[a], xb = x[b];
                x[a] = xa * cos[i] - xb * sin[i];
                x[b] = xa * sin[i] + xb * cos[i];
            }
            return;
        }
        for (int i = 0; i < half; i++) {
            int a = i, b = i + half;
            float xa = x[a], xb = x[b];
            x[a] = xa * cos[i] - xb * sin[i];
            x[b] = xa * sin[i] + xb * cos[i];
        }
    }

    void rms_norm_rows(Slice dst_s, CSlice src_s, CSlice w_s,
                       size_t rows, size_t n, size_t stride, float eps) override {
        float* dst = at(dst_s);
        const float* src = at(src_s);
        const float* w = at(w_s);
        spread(rows, [&](size_t r) {
            rms_norm_raw(dst + r * stride, src + r * stride, w, n, eps);
        });
    }

    void norm_rope_rows(Slice x_s, size_t rows, size_t stride, size_t heads,
                        CSlice w_s, float eps, CSlice cos_s, CSlice sin_s,
                        size_t half, const uint32_t* pos) override {
        float* x = at(x_s);
        const float* w = at(w_s);
        const float* cos = at(cos_s);
        const float* sin = at(sin_s);
        const size_t head_dim = half * 2;
        spread(rows, [&](size_t r) {
            float* row = x + r * stride;
            const float* c = cos + (size_t)pos[r] * half;
            const float* s = sin + (size_t)pos[r] * half;
            for (size_t h = 0; h < heads; h++) {
                float* head = row + h * head_dim;
                rms_norm_raw(head, head, w, head_dim, eps);
                rope_raw(head, c, s, (int)half);
            }
        });
    }

    void gather_rows(Slice dst_s, CSlice src_s, size_t width,
                     const uint32_t* rows, size_t count) override {
        if (count && !rows) throw std::runtime_error("backend: gather without rows");
        if (width && count > (size_t)-1 / width / sizeof(float))
            throw std::runtime_error("backend: gather size overflows");
        float* dst = at(dst_s);
        const float* src = at(src_s);
        span(*dst_s.buffer, dst_s.offset * sizeof(float), count * width * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const size_t r = rows[i];
            if (width && r > (size_t)-1 / width / sizeof(float) - src_s.offset)
                throw std::runtime_error("backend: gather row outside the allocation");
            span(*src_s.buffer, (src_s.offset + r * width) * sizeof(float), width * sizeof(float));
            std::memcpy(dst + i * width, src + r * width, width * sizeof(float));
        }
    }

    void silu_mul(Slice dst_s, CSlice gate_s, CSlice up_s, size_t n) override {
        float* dst = at(dst_s);
        const float* gate = at(gate_s);
        const float* up = at(up_s);
        // std::exp per element, matching the scalar form this replaced: a vectorized approximation would shift logits and is a separate change with its own correctness gate.
        chunk(n, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++)
                dst[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
        });
    }

    void add(Slice dst_s, CSlice src_s, size_t n) override {
        float* dst = at(dst_s);
        const float* src = at(src_s);
        chunk(n, [&](size_t begin, size_t end) {
            size_t i = begin;
            if (avx2_) {
                for (; i + 8 <= end; i += 8)
                    _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i),
                                                            _mm256_loadu_ps(src + i)));
            }
            for (; i < end; i++) dst[i] += src[i];
        });
    }

    void route_experts(CSlice scores_s, size_t rows, size_t n_expert, size_t k, bool normalize,
                       Slice ids_s, Slice weights_s) override {
        if (!k || k > n_expert) throw std::runtime_error("backend: routing selects no expert or more than exist");
        if (k > 256) throw std::runtime_error("backend: more than 256 experts per token");
        if (!rows) return;
        span(*scores_s.buffer, scores_s.offset * sizeof(float), size_mul(rows, n_expert) * sizeof(float));
        span(*ids_s.buffer, ids_s.offset * sizeof(float), size_mul(rows, k) * sizeof(float));
        span(*weights_s.buffer, weights_s.offset * sizeof(float), size_mul(rows, k) * sizeof(float));
        const float* scores = at(scores_s);
        float* ids = at(ids_s);
        float* weights = at(weights_s);
        spread(rows, [&](size_t r) {
            const float* s = scores + r * n_expert;
            std::vector<float> p(s, s + n_expert);
            float top = p[0];
            for (size_t e = 1; e < n_expert; ++e) top = std::max(top, p[e]);
            float sum = 0.0f;
            for (float& v : p) { v = std::exp(v - top); sum += v; }
            for (float& v : p) v /= sum;
            // Probabilities are never negative, so a taken expert is marked below every candidate.
            float chosen[256], total = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                size_t best = 0;
                for (size_t e = 1; e < n_expert; ++e)
                    if (p[e] > p[best]) best = e;
                chosen[j] = p[best];
                total += p[best];
                const uint32_t id = (uint32_t)best;
                std::memcpy(ids + r * k + j, &id, sizeof(id));
                p[best] = -1.0f;
            }
            for (size_t j = 0; j < k; ++j) weights[r * k + j] = normalize ? chosen[j] / total : chosen[j];
        });
    }

    void matmul_experts(std::initializer_list<Projection> projections, CSlice X_s, size_t nin,
                        size_t nrows, const Routing& routing, RowRuns runs = {}) override {
        const size_t entries = size_mul(nrows, routing.k);
        if (!entries) return;
        const Grouping g = group_by_expert(routing, entries);
        span(*X_s.buffer, X_s.offset * sizeof(float), size_mul(nrows, nin) * sizeof(float));
        const std::vector<char> decode = decode_rows(nrows, runs);
        // Every projection of the call reads the same rows, quantized once for all of them.
        xq8_.reset(at(X_s), nrows, nin);
        for (const Projection& p : projections) {
            if (!p.data.buffer) throw std::runtime_error("backend: projection without storage");
            span(*p.out.buffer, p.out.offset * sizeof(float), size_mul(entries, p.rows) * sizeof(float));
            expert_products(p.type, p.data, routing.n_expert, at(X_s), nrows, routing.k, routing.k, at(p.out), nin, p.rows, g, decode);
        }
    }

    // The slots' products land in scratch; each row's weighted sum is then formed in slot order and added, the order the interface fixes.
    void matmul_experts_add(uint32_t type, CSlice data, CSlice X_s, Slice Y_s, size_t nin, size_t nout,
                            size_t nrows, const Routing& routing, RowRuns runs = {}) override {
        const size_t k = routing.k, entries = size_mul(nrows, k);
        if (!entries) return;
        const Grouping g = group_by_expert(routing, entries);
        span(*X_s.buffer, X_s.offset * sizeof(float), size_mul(entries, nin) * sizeof(float));
        span(*Y_s.buffer, Y_s.offset * sizeof(float), size_mul(nrows, nout) * sizeof(float));
        span(*routing.weights.buffer, routing.weights.offset * sizeof(float), entries * sizeof(float));
        expert_out_.resize(size_mul(entries, nout));
        xq8_.reset(at(X_s), entries, nin);
        expert_products(type, data, routing.n_expert, at(X_s), entries, 1, k, expert_out_.data(), nin, nout, g,
                        decode_rows(nrows, runs));
        const float* w = at(routing.weights);
        float* Y = at(Y_s);
        spread(nrows, [&](size_t r) {
            for (size_t o = 0; o < nout; ++o) {
                float s = 0.0f;
                for (size_t j = 0; j < k; ++j) s += w[r * k + j] * expert_out_[(r * k + j) * nout + o];
                Y[r * nout + o] += s;
            }
        });
    }

private:
    // A routing's entries by expert: entry order within each expert, experts in id order.
    struct Grouping {
        std::vector<uint32_t> order;
        std::vector<size_t> start;   // n_expert + 1 offsets into order
    };
    Grouping group_by_expert(const Routing& routing, size_t entries) const {
        span(*routing.ids.buffer, routing.ids.offset * sizeof(float), entries * sizeof(float));
        const float* raw = at(routing.ids);
        Grouping g;
        g.start.assign(routing.n_expert + 1, 0);
        std::vector<uint32_t> ids(entries);
        std::memcpy(ids.data(), raw, entries * sizeof(uint32_t));
        for (uint32_t id : ids) {
            if (id >= routing.n_expert) throw std::runtime_error("backend: routed expert out of range");
            ++g.start[id + 1];
        }
        for (size_t e = 0; e < routing.n_expert; ++e) g.start[e + 1] += g.start[e];
        g.order.resize(entries);
        std::vector<size_t> next(g.start.begin(), g.start.end() - 1);
        for (size_t i = 0; i < entries; ++i) g.order[next[ids[i]]++] = (uint32_t)i;
        return g;
    }

    static bool is_kquant(uint32_t type) {
        return type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K;
    }

    // Whether a type's rows meet quantized activations: the decode dots for a generated token's rows, the prompt dots for a prompt's.
    bool quantized_dots(uint32_t type, size_t nin) const { return decode8_ && avx2_ && q8::has_dot(type) && nin % 32 == 0; }

    // The quantized activations for a type, the rows split across the pool when there are enough of them.
    void prepare_x(uint32_t type) {
        xq8_.prepare(type, [this](size_t rows, const auto& fn) {
            const size_t nt = (size_t)std::max(threads_, 1);
            if (nt <= 1 || rows * xq8_.nin < nt * 16384) { fn(size_t(0), rows); return; }
            const size_t chunk = (rows + nt - 1) / nt;
            run_parallel([&](int w) {
                const size_t r0 = std::min(rows, (size_t)w * chunk), r1 = std::min(rows, r0 + chunk);
                if (r0 < r1) fn(r0, r1);
            });
        });
    }

    // Whether each token row of a call is a generated token's, from its runs as matmul reads them.
    static std::vector<char> decode_rows(size_t nrows, RowRuns runs) {
        std::vector<char> decode(nrows, 0);
        each_run(nrows, runs, [&](size_t first, size_t count, bool d) {
            for (size_t r = first; r < first + count; ++r) decode[r] = d;
        });
        return decode;
    }

    // out[e*nout ..] = expert id(e)'s matrix times X row e / per, X having xrows rows and entry e belonging to token row e / k.
    // A generated token's entries take the decode dots, all of a call's in one pool dispatch.
    // A prompt's entries meet their expert's rows through the prompt dots where the type has them (q8_dots.hpp), an expert's entries as the columns of one block so its weights are unpacked once for all of them, and otherwise through one batched float matmul per expert.
    // Either way an entry computes the same whatever else is routed beside it.
    void expert_products(uint32_t type, CSlice data_s, size_t n_expert, const float* X, size_t xrows, size_t per, size_t k,
                         float* out, size_t nin, size_t nout, const Grouping& g, const std::vector<char>& decode) {
        const size_t row_bytes = row_bytes_of(type, nin), stride = size_mul(nout, row_bytes);
        span(*data_s.buffer, data_s.offset * sizeof(float), size_mul(n_expert, stride));
        const uint8_t* data = (const uint8_t*)bytes_at(data_s);
        const bool q8 = quantized_dots(type, nin);
        if (q8) {
            if (xq8_.src != X || xq8_.rows != xrows || xq8_.nin != nin) xq8_.reset(X, xrows, nin);
            prepare_x(type);
        }
        std::vector<uint32_t> single, expert_of;
        std::vector<size_t> first(n_expert + 1, 0);
        expert_rows_.clear();
        expert_outs_.clear();
        for (size_t e = 0; e < n_expert; ++e) {
            first[e] = expert_rows_.size();
            for (size_t c = g.start[e]; c < g.start[e + 1]; ++c) {
                const uint32_t i = g.order[c];
                if (decode[i / k]) {
                    single.push_back(i);
                    expert_of.push_back((uint32_t)e);
                } else if (q8) {
                    expert_rows_.push_back(i / per);
                    expert_outs_.push_back(out + (size_t)i * nout);
                }
            }
        }
        first[n_expert] = expert_rows_.size();
        if (!single.empty()) {
            const size_t rows = single.size() * nout;
            auto work = [&](size_t r0, size_t r1) {
                for (size_t r = r0; r < r1; ++r) {
                    const size_t s = r / nout, o = r - s * nout, i = single[s];
                    const uint8_t* w = data + expert_of[s] * stride + o * row_bytes;
                    out[i * nout + o] = q8 ? q8::dot(type, w, xq8_, i / per) : row_dot(type, w, X + (i / per) * nin, nin);
                }
            };
            const size_t nt = (size_t)std::max(threads_, 1);
            if (nt <= 1 || rows < nt * 8) work(0, rows);
            else {
                const size_t chunk = (rows + nt - 1) / nt;
                run_parallel([&](int w) { work(std::min(rows, (size_t)w * chunk), std::min(rows, (size_t)(w + 1) * chunk)); });
            }
        }
        if (q8) {
            // Stretches of an expert's rows, handed out as workers free up, since experts carry different numbers of entries.
            const size_t R = 16, per_expert = (nout + R - 1) / R;
            std::vector<uint32_t> busy;
            for (size_t e = 0; e < n_expert; ++e)
                if (first[e + 1] > first[e]) busy.push_back((uint32_t)e);
            const size_t tasks = busy.size() * per_expert;
            if (!tasks) return;
            std::atomic<size_t> next{0};
            auto work = [&](int) {
                for (size_t t = next.fetch_add(1); t < tasks; t = next.fetch_add(1)) {
                    const size_t e = busy[t / per_expert], o0 = (t % per_expert) * R, o1 = std::min(nout, o0 + R);
                    q8::dot_block(type, data + e * stride + o0 * row_bytes, row_bytes, o1 - o0, xq8_, expert_rows_.data() + first[e],
                                  first[e + 1] - first[e], expert_outs_.data() + first[e], o0);
                }
            };
            if (threads_ <= 1 || (first[n_expert] * nout) < (size_t)threads_ * 64) work(0);
            else run_parallel(work);
            return;
        }
        for (size_t e = 0; e < n_expert; ++e) {
            std::vector<uint32_t> batch;
            for (size_t c = g.start[e]; c < g.start[e + 1]; ++c)
                if (!decode[g.order[c] / k]) batch.push_back(g.order[c]);
            if (batch.empty()) continue;
            const uint8_t* w = data + e * stride;
            expert_x_.resize(batch.size() * nin);
            expert_y_.resize(batch.size() * nout);
            for (size_t c = 0; c < batch.size(); ++c)
                std::memcpy(expert_x_.data() + c * nin, X + (batch[c] / per) * nin, nin * sizeof(float));
            matmul_raw(type, w, expert_x_.data(), expert_y_.data(), nin, nout, batch.size(), false);
            for (size_t c = 0; c < batch.size(); ++c)
                std::memcpy(out + (size_t)batch[c] * nout, expert_y_.data() + c * nout, nout * sizeof(float));
        }
    }

    // One weight row against one activation row: the fused dots where a type has one, and for the K-quants the dequantized dot when a fused sum overflows.
    float row_dot(uint32_t type, const uint8_t* row, const float* x, size_t nin) {
        switch (type) {
        case gguf::GGML_TYPE_F32: return dot_f32((const float*)row, x, nin);
        case gguf::GGML_TYPE_Q8_0: return dot_row_impl(row, x, nin / gguf::Q8_0_BLOCK);
        case gguf::GGML_TYPE_Q4_K: case gguf::GGML_TYPE_Q5_K: case gguf::GGML_TYPE_Q6_K: {
            const size_t nb = nin / gguf::Q4_K_BLOCK;
            const float v = type == gguf::GGML_TYPE_Q4_K ? dot_row_q4_K(row, x, nb)
                          : type == gguf::GGML_TYPE_Q5_K ? dot_row_q5_K(row, x, nb) : dot_row_q6_K(row, x, nb);
            return std::isfinite(v) ? v : dot_row_dequant(type, row, x, nin, nb);
        }
        default: {
            const quant::QuantType* qt = quant::Registry::instance().get(type);
            return dot_row_dequant(type, row, x, nin, nin / qt->block_size);
        }
        }
    }

    static size_t size_mul(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a)
            throw std::runtime_error("backend: expert operand size overflows");
        return a * b;
    }

    static size_t row_bytes_of(uint32_t type, size_t nin) {
        if (type == gguf::GGML_TYPE_F32) return size_mul(nin, sizeof(float));
        const quant::QuantType* qt = quant::Registry::instance().get(type);
        if (!qt || !qt->block_size || nin % qt->block_size)
            throw std::runtime_error("backend: expert matrix type or width unsupported");
        return nin / qt->block_size * qt->type_size;
    }

    std::vector<float> expert_x_, expert_y_, expert_out_;
    std::vector<size_t> expert_rows_;   // each grouped entry's activation row
    std::vector<float*> expert_outs_;   // and where its products go
    std::vector<size_t> prompt_rows_;   // a prompt matmul's columns
    std::vector<float*> prompt_outs_;   // and where each column's products go
    q8::Activations xq8_;      // the quantized activations of the last decode call
    bool decode8_ = true;
    // A slice resolves to a host pointer exactly once per op; the kernels below are untouched and still see plain float arrays.
    static float* at(Slice s) {
        if (!s.buffer) throw std::runtime_error("backend: operand without storage");
        // An empty allocation has no address and nothing will read through it;
        // a zero-length batch reaches here.
        if (!s.buffer->size()) return nullptr;
        return (float*)host(*s.buffer) + s.offset;
    }
    // Weights are not float arrays, so their slice resolves to bytes.
    // The offset is still in floats for one reason: every activation is float and a weight slice always starts at zero.
    static const void* bytes_at(CSlice s) {
        if (!s.buffer) throw std::runtime_error("backend: operand without storage");
        const void* p = s.buffer->host_ptr();
        if (!p) throw std::runtime_error("backend: operand is not host addressable");
        return (const uint8_t*)p + s.offset * sizeof(float);
    }

    static const float* at(CSlice s) {
        if (!s.buffer) throw std::runtime_error("backend: operand without storage");
        if (!s.buffer->size()) return nullptr;
        const void* p = s.buffer->host_ptr();
        if (!p) throw std::runtime_error("backend: operand is not host addressable");
        return (const float*)p + s.offset;
    }

    static void* host(const Buffer& b) {
        void* p = dynamic_cast<const CpuBuffer&>(b).host_address();
        if (!p) throw std::runtime_error("backend: buffer is not host addressable");
        return p;
    }
    static void span(const Buffer& b, size_t off, size_t bytes) {
        if (off > b.size() || bytes > b.size() - off)
            throw std::runtime_error("backend: buffer range outside the allocation");
    }

    static CpuKVStorage& storage_of(const KVView& view) {
        auto* s = dynamic_cast<CpuKVStorage*>(view.storage);
        if (!s) throw std::runtime_error("backend: KV view does not belong to the CPU backend");
        return *s;
    }

    // Row-wise dispatch.
    // Below two rows per worker the dispatch costs more than it saves.
    template <typename Fn>
    void spread(size_t rows, const Fn& fn) {
        if (threads_ <= 1 || rows < (size_t)threads_ * 2) {
            for (size_t r = 0; r < rows; r++) fn(r);
            return;
        }
        parallel_for((int)rows, [&](int r) { fn((size_t)r); });
    }

    // Element-wise dispatch over contiguous spans.
    // The threshold keeps decode (where n is a few thousand) on the calling thread.
    template <typename Fn>
    void chunk(size_t n, const Fn& fn) {
        const size_t kMinParallel = 1u << 15;
        if (threads_ <= 1 || n < kMinParallel) { fn(0, n); return; }
        const size_t nt = std::min((size_t)threads_, n);
        const size_t span = (n + nt - 1) / nt;
        parallel_for((int)nt, [&](int w) {
            const size_t begin = (size_t)w * span;
            fn(begin, std::min(n, begin + span));
        });
    }

    int threads_ = 1;
    bool avx2_ = false;
    Ticket ticket_ = 0;
    bool f16c_ = false;
    bool prefill_active_ = false;

    // Persistent worker pool.
    // The previous code created and joined std::threads on every matvec call, which is once per matmul per layer per token; at 36 layers that is thousands of thread creations per token.
    std::vector<std::thread> pool_;
    // Per-worker dequantized weight-row scratch for the batched matmul path.
    std::vector<std::vector<float>> rowbuf_;
    std::vector<float> attention_scores_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_done_;
    // A decode token issues roughly 196 matvecs plus 28 attention calls, each a full dispatch.
    // Measured at 14.1 us per empty dispatch through the condition variable alone, that is a fixed cost of several ms per token.
    // Workers and the caller therefore spin briefly before blocking: the common case is that the other side is already running and arrives within a few hundred nanoseconds.
    // The count is bounded so an idle pool still parks instead of burning a core.
    static const int SPIN_LIMIT = 2048;
    // epoch_ and stop_ are read by the spin loops WITHOUT the mutex, so they must be atomic.
    // Writers still modify them under it; the atomics exist for the unsynchronized readers.
    // Leaving them plain is a data race, and not only a formal one: nothing in a spin body writes them and _mm_pause() is no barrier against another thread, so a compiler may hoist the loads out of the loop and spin forever.
    const std::function<void(int)>* job_ = nullptr;
    std::exception_ptr worker_error_;
    std::atomic<unsigned> epoch_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};

    void start_pool() {
        try {
            rowbuf_.assign((size_t)(threads_ > 0 ? threads_ : 1), std::vector<float>());
            stop_.store(false);
            epoch_.store(0);
            pending_.store(0);
            for (int i = 1; i < threads_; i++)
                pool_.emplace_back([this, i] { worker(i); });
        } catch (...) {
            stop_pool();
            threads_ = 1;
            throw;
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            epoch_++;
        }
        cv_work_.notify_all();
        for (auto& t : pool_) if (t.joinable()) t.join();
        pool_.clear();
    }

    void worker(int idx) {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(int)>* job = nullptr;
            for (int i = 0; i < SPIN_LIMIT && epoch_ == seen && !stop_; i++)
                _mm_pause();
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [&] { return stop_ || epoch_ != seen; });
                if (stop_) return;
                seen = epoch_;
                job = job_;
            }
            std::exception_ptr error;
            try { if (job) (*job)(idx); }
            catch (...) { error = std::current_exception(); }
            if (error) {
                std::lock_guard<std::mutex> lk(m_);
                if (!worker_error_) worker_error_ = error;
            }
            // Notify under the lock so a caller that has just decided to block cannot miss the wakeup.
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m_);
                cv_done_.notify_one();
            }
        }
    }

    // F16C (hardware half<->float).
    // Present on every AVX2 part in practice, but detected separately because the ISA bits are independent.
    static bool has_f16c() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 1);
        return (info[2] & (1 << 29)) != 0;   // ECX bit 29 = F16C
#elif defined(__GNUC__) || defined(__clang__)
        unsigned eax, ebx, ecx, edx;
        if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
        return (ecx & (1u << 29)) != 0;
#else
        return false;
#endif
    }

    static bool has_avx2() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        int maxid = info[0];
        if (maxid < 7) return false;
        __cpuid(info, 7);
        return (info[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
#elif defined(__GNUC__) || defined(__clang__)
        unsigned eax, ebx, ecx, edx;
        if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return false;
        if (eax < 7) return false;
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & (1u << 5)) != 0; // EBX bit 5 = AVX2
#else
        return false;
#endif
    }

    // Fused K-quant row dots, with no dequantized value materialised.
    // A Q4_K sub-block value is d*q - m, so its dot is:
    //     sum_l (d*q_l - m) * x_l  =  d * sum_l(q_l * x_l)  -  m * sum_l(x_l)
    // Q6_K has signed values and per-16 group scales but no min: the sum over groups of (d * sc_g) * sum(q*x). A 128-value chunk holds four sub-blocks of 32, each two 16-value groups with scales sc[2k] and sc[2k+1].
    float dot_row_q6_K(const uint8_t* row, const float* x, size_t nblocks) {
        float acc = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t* p = row + b * gguf::Q6_K_TYPESIZE;
            const uint8_t* ql = p;
            const uint8_t* qh = p + 128;
            const int8_t*  sc = (const int8_t*)(p + 192);
            const float d = half_to_float((uint16_t)(p[208] | ((uint16_t)p[209] << 8)));
            const float* xp = x + b * gguf::Q6_K_BLOCK;

            for (int n = 0; n < (int)gguf::Q6_K_BLOCK; n += 128) {
                for (int k = 0; k < 4; k++) {
                    const uint8_t* qlk = ql + ((k & 1) ? 32 : 0);
                    const bool high = k >= 2;
                    const int shift = 2 * k;
                    const float* xk = xp + k * 32;
                    for (int is = 0; is < 2; is++) {
                        float s;
                        if (avx2_) {
                            // Shifting 16-bit lanes leaks neighbouring bits into the high half of each byte; the mask drops them, so the kept bits are this byte's own.
                            const __m128i cnt = _mm_cvtsi32_si128(shift);
                            const __m128i rawl = _mm_loadu_si128((const __m128i*)(qlk + is * 16));
                            const __m128i rawh = _mm_loadu_si128((const __m128i*)(qh + is * 16));
                            const __m128i lo = high
                                ? _mm_and_si128(_mm_srli_epi16(rawl, 4), _mm_set1_epi8(0x0F))
                                : _mm_and_si128(rawl, _mm_set1_epi8(0x0F));
                            const __m128i hb = _mm_slli_epi16(
                                _mm_and_si128(_mm_srl_epi16(rawh, cnt), _mm_set1_epi8(3)), 4);
                            const __m128i q = _mm_sub_epi8(
                                _mm_or_si128(lo, _mm_and_si128(hb, _mm_set1_epi8((char)0x30))),
                                _mm_set1_epi8(32));
                            __m256 a = _mm256_mul_ps(
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                                _mm256_loadu_ps(xk + is * 16));
                            a = _mm256_fmadd_ps(
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))),
                                _mm256_loadu_ps(xk + is * 16 + 8), a);
                            s = hsum256(a);
                        } else {
                            s = 0.0f;
                            for (int l = is * 16; l < is * 16 + 16; l++) {
                                const int nib = high ? (qlk[l] >> 4) : (qlk[l] & 0xF);
                                const int qv = (nib | (((qh[l] >> shift) & 3) << 4)) - 32;
                                s += (float)qv * xk[l];
                            }
                        }
                        acc += d * (float)sc[k * 2 + is] * s;
                    }
                }
                ql += 64; qh += 32; sc += 8; xp += 128;
            }
        }
        return acc;
    }

    // Same factorisation as Q4_K: the value is d*(q + 16*hbit) - m, so the dot is d*sum((q + 16*hbit)*x) - m*sum(x) and no dequantized value is materialised.
    // The fifth bit comes from qh, whose mask shifts left by two every 64 values while qh itself does not advance.
    float dot_row_q5_K(const uint8_t* row, const float* x, size_t nblocks) {
        float acc = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t* p = row + b * gguf::Q5_K_TYPESIZE;
            const float d    = half_to_float((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
            const float dmin = half_to_float((uint16_t)(p[2] | ((uint16_t)p[3] << 8)));
            const uint8_t* sc = p + 4;
            const uint8_t* qh = p + 16;
            const uint8_t* ql = p + 48;
            const float* xp = x + b * gguf::Q5_K_BLOCK;

            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < (int)gguf::Q5_K_BLOCK; j += 64) {
                uint8_t s, mm;
                quant::get_scale_min_k4(is + 0, sc, &s, &mm);
                const float d1 = d * (float)s, m1 = dmin * (float)mm;
                quant::get_scale_min_k4(is + 1, sc, &s, &mm);
                const float d2 = d * (float)s, m2 = dmin * (float)mm;

                if (avx2_) {
                    const __m128i lo_mask = _mm_set1_epi8(0x0F);
                    const __m128i sixteen = _mm_set1_epi8(16);
                    const __m128i b1 = _mm_set1_epi8((char)u1);
                    const __m128i b2 = _mm_set1_epi8((char)u2);
                    __m128i rawl[2] = { _mm_loadu_si128((const __m128i*)(ql +  0)),
                                        _mm_loadu_si128((const __m128i*)(ql + 16)) };
                    __m128i rawh[2] = { _mm_loadu_si128((const __m128i*)(qh +  0)),
                                        _mm_loadu_si128((const __m128i*)(qh + 16)) };
                    __m256 qx_lo = _mm256_setzero_ps(), sx_lo = _mm256_setzero_ps();
                    __m256 qx_hi = _mm256_setzero_ps(), sx_hi = _mm256_setzero_ps();
                    for (int h = 0; h < 2; h++) {
                        // A set bit contributes exactly 16 to the value, so compare-then-mask gives the addend without a branch.
                        const __m128i hb = rawh[h];
                        const __m128i add_lo = _mm_and_si128(
                            _mm_cmpeq_epi8(_mm_and_si128(hb, b1), b1), sixteen);
                        const __m128i add_hi = _mm_and_si128(
                            _mm_cmpeq_epi8(_mm_and_si128(hb, b2), b2), sixteen);
                        const __m128i L = _mm_add_epi8(
                            _mm_and_si128(rawl[h], lo_mask), add_lo);
                        const __m128i H = _mm_add_epi8(
                            _mm_and_si128(_mm_srli_epi16(rawl[h], 4), lo_mask), add_hi);
                        for (int q = 0; q < 2; q++) {
                            const __m128i Lp = q ? _mm_srli_si128(L, 8) : L;
                            const __m128i Hp = q ? _mm_srli_si128(H, 8) : H;
                            const __m256 fl = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(Lp));
                            const __m256 fh = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(Hp));
                            const __m256 xl = _mm256_loadu_ps(xp + h * 16 + q * 8);
                            const __m256 xh = _mm256_loadu_ps(xp + 32 + h * 16 + q * 8);
                            qx_lo = _mm256_fmadd_ps(fl, xl, qx_lo);
                            sx_lo = _mm256_add_ps(xl, sx_lo);
                            qx_hi = _mm256_fmadd_ps(fh, xh, qx_hi);
                            sx_hi = _mm256_add_ps(xh, sx_hi);
                        }
                    }
                    acc += d1 * hsum256(qx_lo) - m1 * hsum256(sx_lo);
                    acc += d2 * hsum256(qx_hi) - m2 * hsum256(sx_hi);
                } else {
                    float qx1 = 0, sx1 = 0, qx2 = 0, sx2 = 0;
                    for (int l = 0; l < 32; l++) {
                        const float xa = xp[l], xb = xp[l + 32];
                        qx1 += (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) * xa;
                        sx1 += xa;
                        qx2 += (float)((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) * xb;
                        sx2 += xb;
                    }
                    acc += d1 * qx1 - m1 * sx1;
                    acc += d2 * qx2 - m2 * sx2;
                }
                xp += 64; ql += 32; is += 2;
                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
        }
        return acc;
    }

    // Exact-ish fallback for a row whose fused dot overflowed: dequantize first so the block scale is multiplied into each weight before it meets the activation, which is what keeps intermediates finite.
    // Accumulates in double so the fallback itself cannot overflow where the reference would not.
    // Rare by construction, so a local buffer is cheaper than reserving per-worker scratch that is almost never touched.
    float dot_row_dequant(uint32_t type, const uint8_t* row, const float* x,
                          size_t nin, size_t nb) {
        const quant::QuantType* qt = quant::Registry::instance().get(type);
        if (!qt || !qt->dequantize)
            throw std::runtime_error("backend: no dequantizer for fallback dot");
        std::vector<float> buf(nin);
        qt->dequantize(row, buf.data(), nb);
        double s = 0.0;
        for (size_t i = 0; i < nin; i++) s += (double)buf[i] * (double)x[i];
        return (float)s;
    }

    float dot_row_q4_K(const uint8_t* row, const float* x, size_t nblocks) {
        float acc = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t* p = row + b * gguf::Q4_K_TYPESIZE;
            const float d    = half_to_float((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
            const float dmin = half_to_float((uint16_t)(p[2] | ((uint16_t)p[3] << 8)));
            const uint8_t* sc = p + 4;
            const uint8_t* qs = p + 16;
            const float* xp = x + b * gguf::Q4_K_BLOCK;

            int is = 0;
            for (int j = 0; j < (int)gguf::Q4_K_BLOCK; j += 64) {
                uint8_t s, mm;
                quant::get_scale_min_k4(is + 0, sc, &s, &mm);
                const float d1 = d * (float)s, m1 = dmin * (float)mm;
                quant::get_scale_min_k4(is + 1, sc, &s, &mm);
                const float d2 = d * (float)s, m2 = dmin * (float)mm;

                if (avx2_) {
                    const __m128i lo_mask = _mm_set1_epi8(0x0F);
                    __m128i raw0 = _mm_loadu_si128((const __m128i*)(qs +  0));
                    __m128i raw1 = _mm_loadu_si128((const __m128i*)(qs + 16));
                    // low nibbles -> first 32 values, high nibbles -> next 32
                    __m128i lo0 = _mm_and_si128(raw0, lo_mask);
                    __m128i lo1 = _mm_and_si128(raw1, lo_mask);
                    __m128i hi0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), lo_mask);
                    __m128i hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), lo_mask);

                    __m256 qx_lo = _mm256_setzero_ps(), sx_lo = _mm256_setzero_ps();
                    __m256 qx_hi = _mm256_setzero_ps(), sx_hi = _mm256_setzero_ps();
                    const __m128i* lohalves[2] = { &lo0, &lo1 };
                    const __m128i* hihalves[2] = { &hi0, &hi1 };
                    for (int h = 0; h < 2; h++) {
                        const __m128i L = *lohalves[h];
                        const __m128i H = *hihalves[h];
                        for (int q = 0; q < 2; q++) {
                            const __m128i Lp = q ? _mm_srli_si128(L, 8) : L;
                            const __m128i Hp = q ? _mm_srli_si128(H, 8) : H;
                            const __m256 fl = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(Lp));
                            const __m256 fh = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(Hp));
                            const __m256 xl = _mm256_loadu_ps(xp + h * 16 + q * 8);
                            const __m256 xh = _mm256_loadu_ps(xp + 32 + h * 16 + q * 8);
                            qx_lo = _mm256_fmadd_ps(fl, xl, qx_lo);
                            sx_lo = _mm256_add_ps(xl, sx_lo);
                            qx_hi = _mm256_fmadd_ps(fh, xh, qx_hi);
                            sx_hi = _mm256_add_ps(xh, sx_hi);
                        }
                    }
                    acc += d1 * hsum256(qx_lo) - m1 * hsum256(sx_lo);
                    acc += d2 * hsum256(qx_hi) - m2 * hsum256(sx_hi);
                } else {
                    float qx1 = 0, sx1 = 0, qx2 = 0, sx2 = 0;
                    for (int l = 0; l < 32; l++) {
                        const float xa = xp[l], xb = xp[l + 32];
                        qx1 += (float)(qs[l] & 0xF) * xa; sx1 += xa;
                        qx2 += (float)(qs[l] >> 4)  * xb; sx2 += xb;
                    }
                    acc += d1 * qx1 - m1 * sx1;
                    acc += d2 * qx2 - m2 * sx2;
                }
                xp += 64; qs += 32; is += 2;
            }
        }
        return acc;
    }

    static float hsum256(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }

    // f16 -> f32 using hardware F16C where present.
    float half_to_float(uint16_t h) const {
        if (f16c_) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
        return f16_to_f32(h);
    }

    // Dot product of one Q8_0 row (nblocks blocks, nin = nblocks*32) with x.
    // Uses an AVX2 fused dequant+FMA path when available, else scalar.
    float dot_row_impl(const uint8_t* row, const float* x, size_t nblocks) {
        if (avx2_) {
            // Four independent accumulators.
            // A single chained accumulator serialised the loop at FMA latency, which also capped how many loads could be in flight; decode is bandwidth bound, so fewer outstanding loads means less memory-level parallelism and less achieved bandwidth.
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            // No software prefetch: the hardware prefetcher already keeps up with these sequential streams.
            for (size_t b = 0; b < nblocks; b++) {
                const uint8_t* y = row + b * gguf::Q8_0_TYPESIZE;
                // Hardware f16 convert.
                // The scalar f16_to_f32 is a branchy function (zero, subnormal, inf/nan cases) called once per 34 bytes of weights, which is a lot of unpredictable control flow in a loop whose job is to keep loads in flight.
                __m256 dv;
                if (f16c_) {
                    dv = _mm256_cvtph_ps(_mm_broadcastw_epi16(_mm_loadu_si128((const __m128i*)y)));
                } else {
                    const uint16_t h = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
                    dv = _mm256_set1_ps(f16_to_f32(h));
                }
                __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(y + 2))));
                __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(y + 10))));
                __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(y + 18))));
                __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(y + 26))));
                const float* xp = x + b * gguf::Q8_0_BLOCK;
                s0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, dv), _mm256_loadu_ps(xp), s0);
                s1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, dv), _mm256_loadu_ps(xp + 8), s1);
                s2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, dv), _mm256_loadu_ps(xp + 16), s2);
                s3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, dv), _mm256_loadu_ps(xp + 24), s3);
            }
            __m256 acc = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        }
        float acc = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t* y = row + b * gguf::Q8_0_TYPESIZE;
            uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
            float d = f16_to_f32(d16);
            for (int j = 0; j < gguf::Q8_0_BLOCK; j++)
                acc += (float)(int8_t)y[2 + j] * d * x[b * gguf::Q8_0_BLOCK + j];
        }
        return acc;
    }
};

// Default CPU backend factory.
inline BackendPtr make_cpu_backend() { return std::make_shared<CpuBackend>(); }

} // namespace backend
