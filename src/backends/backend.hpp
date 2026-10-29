#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Backends own storage and parallelize primitive ops; models use buffer handles and own multi-device placement.
// The execution and ownership contracts are in docs/DEVICE-EXECUTION.md.

namespace backend {

// Storage owned by the backend that allocated it.
// Models pass buffer handles to compute ops; host-visible results can be read after retirement.
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    // Non-null for nonempty host-addressable allocations: Memory::host_visible and all host-backend storage.
    // What it points at is current only after wait() or sync().
    virtual const void* host_ptr() const = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

// Throws unless `bytes` bytes from byte `off` lie inside the buffer.
inline void span(const Buffer& b, size_t off, size_t bytes) {
    if (off > b.size() || bytes > b.size() - off)
        throw std::runtime_error("backend: buffer range outside the allocation");
}

// Size arithmetic for storage and operands: a product or sum that would wrap throws instead.
inline size_t size_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error("backend: size overflows");
    return a * b;
}
inline size_t size_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error("backend: size overflows");
    return a + b;
}

// device memory may be unreachable from the host; host_visible permits host_ptr() access after retirement.
// Both use the same storage on a host backend.
enum class Memory { device, host_visible };

// A submission.
// `submit()` hands everything enqueued so far to the device and returns one; `wait()` blocks until that submission has retired.
using Ticket = uint64_t;

// An operand identifies backend storage with a buffer and a float offset.
struct Slice {
    Buffer* buffer = nullptr;
    size_t offset = 0;
};
struct CSlice {
    const Buffer* buffer = nullptr;
    size_t offset = 0;
    CSlice() = default;
    CSlice(const Buffer* b, size_t o) : buffer(b), offset(o) {}
    CSlice(const Slice& s) : buffer(s.buffer), offset(s.offset) {}
};

// int8 rounds the inputs of matrix products to 8 bits (docs/PRECISION.md): a speed option below the default precision, which auto never chooses.
enum class Dtype { f32, f16, bf16, int8 };
enum class MatrixPath { f32, f16, bf16, block_int16, block_int8 };

// The dispatched matrix forms, accumulated without allocating; read after the work has completed.
class MatrixPaths {
public:
    void record(MatrixPath path) { bits_ |= 1u << unsigned(path); }
    std::vector<std::string> take() {
        static const char* const names[] = {"f32", "f16", "bf16", "block-int16", "block-int8"};
        std::vector<std::string> paths;
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
            if (bits_ & (1u << i)) paths.emplace_back(names[i]);
        bits_ = 0;
        return paths;
    }
private:
    uint32_t bits_ = 0;
};
inline const char* dtype_name(Dtype dtype) {
    switch (dtype) {
        case Dtype::f32: return "f32";
        case Dtype::f16: return "f16";
        case Dtype::bf16: return "bf16";
        case Dtype::int8: return "int8";
    }
    return "unknown";
}

// A matrix or embedding operand, with optional F32 scalar/per-expert weight scales.
struct WeightView {
    uint32_t type;
    CSlice data;
    CSlice scales = {};
    size_t scale_count = 0;
};

struct Projection {
    WeightView weight;
    Slice out;
    size_t rows;
    Projection() = default;
    // Keep the weight operand grouped: the old flat layout would fill its scale fields.
    Projection(WeightView w, Slice y, size_t n) : weight(w), out(y), rows(n) {}
};

// The rows of a matmul grouped by the prompt they belong to, so a device picks a row's kernel by `extent` rather than by the call's width and a prompt computes the same however its rows are batched (docs/VULKAN.md, batch invariance).
// `end` is one past the run's last row and `extent` the position one past the prompt's last token, 1 for a generated token; without runs a backend chooses by the call's width.
struct RowRun {
    size_t end;
    size_t extent;
};
struct RowRuns {
    const RowRun* runs = nullptr;
    size_t n = 0;
};

// A routed feed-forward step's operands (Backend::moe_step): the shared expert's, beside the routed experts' below.
// Its gate and up project one row a token into `out`, its row gate, an F32 vector, gives one value a row and has no data on an ungated expert.
struct MoeShared {
    Projection gate, up;
    WeightView down;
    Slice act;
    Projection row_gate;
};
// The F32 router's n_expert scores into router.out, a row's k picks and weights, each stack's n_expert matrices of gate.rows rows writing k rows a token, and the slot the SiLU writes.
// `zero`, a row of zeros as wide as y's rows, clears y before the sums join it; `scratch` holds the entry runs the routed SiLU takes.
struct MoeStep {
    Projection router;
    size_t k;
    bool normalize;
    Slice ids, weights;
    Projection gate, up;
    WeightView down;
    Slice act;
    const MoeShared* shared;
    const Buffer* zero;
    std::vector<RowRun>* scratch;
};

// Calls `each(first, count, key)` over the stretches of adjacent runs whose `key(run)` is equal, once the runs are known to be in row order and to end at row `n`.
// Every run is checked before the first call, so a malformed list reaches no rows; a caller handles a call without runs itself.
template <typename Key, typename Each>
void for_each_run(size_t n, RowRuns runs, const Key& key, const Each& each) {
    size_t end = 0;
    for (size_t i = 0; i < runs.n; ++i) {
        if (runs.runs[i].end < end) throw std::runtime_error("backend: row runs out of order");
        end = runs.runs[i].end;
    }
    if (end != n) throw std::runtime_error("backend: row runs do not cover the batch");
    size_t start = 0;
    for (size_t i = 0; i < runs.n;) {
        const auto k = key(runs.runs[i]);
        size_t j = i + 1;
        while (j < runs.n && key(runs.runs[j]) == k) ++j;
        const size_t stop = runs.runs[j - 1].end;
        if (stop > start) each(start, stop - start, k);
        start = stop;
        i = j;
    }
}

// The backend owns KV storage and its block layout; models keep logical views (docs/KV-CACHE.md).
// A block id indexes the same block in every layer of one KVStorage.
struct KVLayout {
    size_t block_tokens;
};

// Whole blocks of `block_tokens` for `tokens` positions, without the usual +bt-1 overflow.
inline size_t blocks_for(size_t tokens, size_t block_tokens) {
    return tokens / block_tokens + (tokens % block_tokens != 0);
}

// How a cache side is stored; the CLI's --cache-type-k and --cache-type-v.
enum class KVType { f32, f16 };
inline size_t kv_elem_bytes(KVType t) { return t == KVType::f16 ? 2 : 4; }
// The same two names on every backend.
inline KVType kv_type_of(const std::string& name) {
    if (name == "f32") return KVType::f32;
    if (name == "f16") return KVType::f16;
    throw std::runtime_error("unknown cache type '" + name + "' (f32 or f16)");
}
inline const char* kv_type_name(KVType t) { return t == KVType::f16 ? "f16" : "f32"; }

class KVStorage {
public:
    virtual ~KVStorage() = default;
    virtual size_t max_blocks() const = 0;
    // Bytes retained for blocks now, and the most held across successful growths (a growth copies, so old plus new is held for a moment).
    // A growth that failed part way is not counted.
    virtual size_t allocated_bytes() const = 0;
    virtual size_t peak_bytes() const = 0;
    // Backs every block of the budget at once, so no later write grows the storage.
    virtual void back_all() = 0;
};

// One sequence's history in one storage: logical block i is physical block blocks[i], `length` entries are committed, and `nq` rows of this pass belong to the sequence.
// Row b is at position length + b and attends through it, so the table must cover length + nq; `extent` is the rows' RowRun extent, 0 when unknown.
struct KVView {
    KVStorage* storage;
    const int32_t* blocks;
    size_t n_blocks;
    size_t length;
    size_t nq;
    size_t extent = 0;
    // A non-causal view's rows attend every position of the view, length + nq, as a vision encoder's rows see their whole image (docs/VISION.md, section 3).
    bool causal = true;
};

// Several positions a row for a rope: row r's stream k is pos[r * count + k], and frequency j turns by stream of[j]'s (docs/VISION.md, section 4).
// One stream, the default, is the plain rope.
struct RopeStreams {
    size_t count = 1;
    const uint8_t* of = nullptr;
};

// The positions the groups a call pools are rotated at: for each view in order, each group its rows complete, `streams.count` positions a group (RopeStreams).
// With none a group turns at its first token's index, as a history without images has it.
struct GroupPositions {
    const uint32_t* pos = nullptr;
    RopeStreams streams{};
};

// GELU as x times the normal CDF of x, or its tanh approximation (Op::gelu).
enum class GeluForm { tanh, erf };

// The width of the linear-attention layers' causal conv, 4 in every qwen35 file; a slot carries the kConvTaps - 1 raw rows before a sequence's next token.
inline constexpr size_t kConvTaps = 4;

// The epsilon of the linear-attention layers' L2 norms of q and k, which no file carries (docs/QWEN35.md, What the converter folds).
inline constexpr float kL2NormEps = 1e-6f;

// One linear-attention layer's recurrent state for one sequence (docs/QWEN35.md, The recurrent state): K heads of k_dim and V heads of v_dim, V head j reading K head j mod k_heads.
struct StateShape {
    size_t k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0;
    // The conv's channels, those of a raw projection row [q: k_heads k_dim | k: k_heads k_dim | v: v_heads v_dim].
    size_t channels() const { return size_add(size_mul(2 * k_heads, k_dim), size_mul(v_heads, v_dim)); }
    // Floats of each V head's matrix, laid out [K row][V column].
    size_t matrix_floats() const { return size_mul(k_dim, v_dim); }
    // Floats of a slot: every V head's matrix, V head by V head, then the conv's kConvTaps - 1 carried rows of channels(), oldest first.
    size_t slot_floats() const {
        return size_add(size_mul(v_heads, matrix_floats()), size_mul(kConvTaps - 1, channels()));
    }
};

// The recurrent state of `layers` linear-attention layers, made whole by Backend::state_alloc and never grown: one buffer per layer, holding every slot back to back, all F32.
// It refuses a buffer that cannot hold its slots, so an op resolving a slot stays inside its buffer.
class StateStorage {
public:
    StateStorage(std::vector<BufferPtr> buffers, size_t slots, const StateShape& shape,
                 const std::vector<size_t>& tail_floats = {})
        : buffers_(std::move(buffers)), slots_(slots), shape_(shape), tail_floats_(tail_floats) {
        if (!tail_floats_.empty() && tail_floats_.size() != buffers_.size())
            throw std::runtime_error("backend: state tail count differs from its layers");
        std::vector<size_t> bytes(buffers_.size());
        for (size_t l = 0; l < buffers_.size(); ++l)
            bytes[l] = size_mul(slots_, slot_bytes(l));
        for (size_t l = 0; l < buffers_.size(); ++l)
            if (!buffers_[l] || buffers_[l]->size() < bytes[l])
                throw std::runtime_error("backend: state buffer smaller than its slots");
    }
    size_t layers() const { return buffers_.size(); }
    size_t slots() const { return slots_; }
    const StateShape& shape() const { return shape_; }
    size_t slot_floats(size_t l) const {
        if (l >= buffers_.size()) throw std::runtime_error("backend: state layer outside the storage");
        return size_add(shape_.slot_floats(), tail_floats_.empty() ? 0 : tail_floats_[l]);
    }
    size_t slot_bytes(size_t l) const { return size_mul(slot_floats(l), sizeof(float)); }
    Buffer& layer(size_t l) const {
        if (l >= buffers_.size()) throw std::runtime_error("backend: state layer outside the storage");
        return *buffers_[l];
    }

private:
    std::vector<BufferPtr> buffers_;
    size_t slots_;
    StateShape shape_;
    std::vector<size_t> tail_floats_;
};

// One sequence's rows of a pass in one StateStorage: `nq` rows, in view order, continue a history of `length` tokens.
// The state is read from slot `src` and written to slot `dst`, the same slot except in a verify, and length 0 reads a zero state whatever `src` holds, so a recycled slot needs no clearing.
struct StateView {
    StateStorage* storage = nullptr;
    size_t src = 0, dst = 0;
    size_t length = 0;
    size_t nq = 0;
};

// Throws unless every view has rows, names a storage holding `layer` with slots inside it and the first view's shape, and writes a slot no other view reads or writes, so the views do not depend on their order.
// Returns the rows of all the views.
inline size_t check_state_views(const StateView* views, size_t n_views, size_t layer) {
    if (n_views && !views) throw std::runtime_error("backend: state op without views");
    size_t rows = 0;
    for (size_t i = 0; i < n_views; ++i) {
        const StateView& v = views[i];
        if (!v.storage) throw std::runtime_error("backend: state view without storage");
        if (layer >= v.storage->layers()) throw std::runtime_error("backend: state layer outside the storage");
        if (v.src >= v.storage->slots() || v.dst >= v.storage->slots())
            throw std::runtime_error("backend: state slot outside the storage");
        if (!v.nq) throw std::runtime_error("backend: state view without rows");
        const StateShape& a = v.storage->shape();
        const StateShape& b = views[0].storage->shape();
        if (a.k_heads != b.k_heads || a.v_heads != b.v_heads || a.k_dim != b.k_dim || a.v_dim != b.v_dim ||
            v.storage->slot_floats(layer) != views[0].storage->slot_floats(layer))
            throw std::runtime_error("backend: state views of different shapes");
        for (size_t j = 0; j < i; ++j) {
            const StateView& u = views[j];
            if (u.storage != v.storage) continue;
            if (u.dst == v.dst || u.dst == v.src || u.src == v.dst)
                throw std::runtime_error("backend: a state slot written by one view is used by another");
        }
        rows = size_add(rows, v.nq);
    }
    return rows;
}

enum class NormGate : uint8_t { silu, sigmoid };
// The ops some backends lack, which a model's plan names for each part that issues one, so a backend without one refuses the model at load rather than in a pass (Backend::implements).
enum class Op : uint8_t { causal_conv_silu, gated_delta_rule, gated_rms_norm, norm_rope_partial, sigmoid_mul, mixed_experts, argmax_rows, embed_ids,
                          group_rms_norm, silu_scaled, mix_streams, inject_streams, signed_sqrt_gate, dilated_conv_silu_add, select_kv, sparse_attention,
                          layer_norm, gelu, add_bias, mix_rows, rope_streams, attention_full };
inline const char* op_name(Op op) {
    switch (op) {
        case Op::causal_conv_silu: return "causal_conv_silu";
        case Op::gated_delta_rule: return "gated_delta_rule";
        case Op::gated_rms_norm: return "gated_rms_norm";
        case Op::norm_rope_partial: return "norm_rope_partial";
        case Op::sigmoid_mul: return "sigmoid_mul";
        case Op::mixed_experts: return "mixed_experts";
        case Op::argmax_rows: return "argmax_rows";
        case Op::embed_ids: return "embed_ids";
        case Op::group_rms_norm: return "group_rms_norm";
        case Op::silu_scaled: return "silu_scaled";
        case Op::mix_streams: return "mix_streams";
        case Op::inject_streams: return "inject_streams";
        case Op::signed_sqrt_gate: return "signed_sqrt_gate";
        case Op::dilated_conv_silu_add: return "dilated_conv_silu_add";
        case Op::select_kv: return "select_kv";
        case Op::sparse_attention: return "sparse_attention";
        case Op::layer_norm: return "layer_norm";
        case Op::gelu: return "gelu";
        case Op::add_bias: return "add_bias";
        case Op::mix_rows: return "mix_rows";
        case Op::rope_streams: return "rope_streams";
        case Op::attention_full: return "attention_full";
    }
    return "an unknown op";
}

// A tensor group's sum over backends of one kind, made once at load (docs/TENSOR-SPLIT.md, section 4.3): sum_into adds the members' partial rows (partial() says where) into every member's residual in member order.
class Collective {
public:
    virtual ~Collective() = default;
    // Where member `member` writes the partial rows of its next sum: storage the collective owns, on that member, for the rows and width it was made for.
    virtual Slice partial(size_t member) = 0;
    // Each member's residual rows, `rows` rows of `width` floats at residual[m], gain ((p0 + p1) + ...) + p(W-1) of the members' partial rows, enqueued on every member.
    virtual void sum_into(const std::vector<Slice>& residual, size_t rows, size_t width) = 0;
    // Every member done with what the last sum submitted on it (`back` 0), or the sum before it (1); nothing to wait for where the members compute as they are called.
    virtual void wait(size_t back) noexcept { (void)back; }
};

// A device's operations and memory; a backend takes one caller at a time, so its calls never come from two threads at once.
class Backend {
public:
    virtual ~Backend() = default;

    // A collective over `members`, this backend first, every one of this backend's kind, for sums of up to `rows` rows of `width` floats; null where this backend has none on this platform.
    virtual std::unique_ptr<Collective> join(const std::vector<Backend*>& members, size_t rows, size_t width) {
        (void)members, (void)rows, (void)width;
        return nullptr;
    }
    // The PCI root complex above the backend's device, as the system names it, which a tensor group's startup report gives; empty where there is none or the system does not say.
    virtual std::string pci_root() const { return {}; }
    // What a member of a collective over `members` members keeps of this backend's memory for each row of `width` floats a sum takes (join), which the fit counts.
    virtual size_t collective_bytes_per_row(size_t members, size_t width) const {
        (void)members, (void)width;
        return 0;
    }

    // Complete native policies, preferred first; emulation alone does not add a policy to auto's choices, and int8, never preferred, is listed last where a backend runs it.
    virtual std::vector<Dtype> native_dtypes() const { return {Dtype::f32}; }
    virtual bool emulates_dtype(Dtype) const { return false; }
    virtual std::string dtype_path(Dtype dtype) const { return dtype_name(dtype); }

    // A model's stage swaps in its own evidence, then restores the caller's on every exit.
    void swap_matrix_paths(MatrixPaths& paths) noexcept { std::swap(matrix_paths_, paths); }


    // Whether the implemented weight-reading ops support this storage type on this device, checked before a model adopts its weights.
    virtual bool supports_type(uint32_t type) const { (void)type; return false; }
    // Whether this backend runs `op`; one that does not refuses it by name when it is called anyway.
    virtual bool implements(Op op) const {
        (void)op;
        return false;
    }

    // While on, the ops recorded need not run in order, the caller's promise that none of them reads or writes what another writes; turned off, everything recorded before runs before what follows.
    // A rerun of many independent state layers records them so (docs/SPECULATIVE.md, section 7); a backend that computes as it records ignores it.
    virtual void unordered(bool on) { (void)on; }

    // The generated tokens a decode pass's products read each weight once for, past which another row costs a weight read of its own: what drafts ride on beside other requests' rows (docs/SPECULATIVE.md, section 3).
    virtual size_t decode_columns() const { return 1; }

    // Set the worker thread count hint; 0 leaves the current count unchanged.
    virtual void set_threads(int n) = 0;

    // Number of worker threads this backend will actually use (after the last set_threads, or auto-detected).
    // Reported by the CLI; a device backend returns whatever is meaningful for it, or 0.
    virtual int threads_available() const = 0;

    // Bytes this backend can still allocate for weights, caches and activations, as the device or the operating system reports them now; nothing when it cannot tell.
    // A placement across several devices is fitted against it (docs/MULTI-DEVICE.md).
    virtual std::optional<size_t> memory_available() const { return std::nullopt; }

    // What adopting a matrix of this quant type and shape keeps resident on this backend: its bytes, and any copy the backend makes of it, so a fit counts it.
    // `product` is a matrix a matrix product reads as its weights, as opposed to a gathered table, a norm or a routed expert stack.
    virtual size_t resident_bytes(uint32_t type, size_t nin, size_t rows, size_t bytes, bool product) const {
        (void)type; (void)nin; (void)rows; (void)product;
        return bytes;
    }

    // Host memory the backend holds for its own use, such as staging for uploads, which counts against the host rather than against memory_available().
    virtual size_t host_resident() const { return 0; }

    // Whether memory_available() is the host's free memory, which the host tier and the rest of the host share, as the CPU's is.
    virtual bool memory_is_host() const { return false; }

    // Whether adopt() reads the caller's memory in place rather than copying it into the backend's own, so weights placed here cost no memory of this backend.
    virtual bool reads_in_place() const { return false; }

    // Whether this backend is the CPU itself, so experts placed on the CPU beside it are already where they run; a device may read in place and still not be the CPU.
    virtual bool is_cpu() const { return false; }

    // What makes this backend's arithmetic its own, as text: with the same build, two backends of one identity give the same bits for the same work at the same row classes (docs/DISK-TIER.md, The entry file).
    // A backend that cannot say gives an identity no other backend shares.
    virtual std::string identity() const { return "unknown " + std::to_string((uintptr_t)this); }

    // The class of rows of this RowRun extent: rows of two extents of one class take the same arithmetic in every op of this backend, so they give the same bits.
    // A backend claims only what it proves, so by default every extent is a class of its own (docs/SPECULATIVE.md, section 1).
    virtual size_t row_class(size_t extent) const { return extent; }

    // Where a backend made to time its work (make_backend's diagnostics) held its caller since it was made, in milliseconds: waiting on tickets, for a free command slot and for staging, and in uploads (write) apart from those waits.
    // A backend that computes as it records never waits, and one not made to time its work reports nothing.
    struct HostTimes {
        double ticket_ms = 0, slot_ms = 0, staging_ms = 0, write_ms = 0;
    };
    virtual HostTimes host_times() const { return {}; }
    // The device time of the work recorded since the last call, in milliseconds, by timestamps on a backend made to time its work, which waits for its queue to read them; negative where the backend keeps none.
    virtual double device_ms() { return -1.0; }

    // Memory the backend's kernels take for themselves beside weights, caches and activations, such as split partials and merge state, out of `free` bytes; a fit keeps it back.
    virtual size_t scratch_reserve(size_t free) const { (void)free; return 0; }

    // Invoke once on the caller and complete all cleanup before returning.
    virtual void run_prefill(const std::function<void()>& work) { work(); }

    // Zero-filled backend storage.
    virtual BufferPtr alloc(size_t bytes, Memory where = Memory::device) = 0;

    // Make src reachable by this backend's ops.
    // A backend that reads in place (reads_in_place) borrows src for the returned buffer's life and does not read it inside adopt; one that copies has consumed src when adopt returns, so the caller may release it then.
    virtual BufferPtr adopt(const void* src, size_t bytes) = 0;

    // Storage for a weight the caller fills with write before any op reads it; it need not be zeroed, and the backend keeps it as it keeps an adopted weight.
    virtual BufferPtr alloc_weight(size_t bytes) { return alloc(bytes); }

    // A buffer over the caller's page-aligned memory that copy reads from in place, or null where this backend cannot read it so.
    // The caller keeps the memory for the buffer's life and leaves it unchanged until the copies out of it retire.
    virtual BufferPtr wrap_host(void* memory, size_t bytes) {
        (void)memory;
        (void)bytes;
        return nullptr;
    }

    // For a device of a placement over several: between its submissions the device waits on itself rather than going idle, so its clock stays up while another device runs its part of a pass (docs/MULTI-DEVICE.md).
    // Each true adds a holder and each false removes one, so a backend kept by several models holds while any asks; a backend without such a wait ignores it, and false must not throw.
    virtual void hold_between_submissions(bool on) { (void)on; }

    // Ops enqueue on one stream; submit() flushes and returns a monotonic ticket, and wait(t) retires that submission and everything before it.
    // Results require wait(), sync() or read(); CPU ops complete eagerly (docs/DEVICE-EXECUTION.md).
    virtual Ticket submit() = 0;
    // Retirement cannot throw, since callers release storage afterward, and a backend is one thread's: every call is by the recording thread, but wait(t) of a submitted ticket may come from another (docs/SERVER.md, the round).
    // A group's queue calls may be made on a thread of their own, which a ticket is waited for on first; a ticket whose call failed there retires without its work.
    virtual void wait(Ticket t) noexcept = 0;
    // After wait(t), before reading in place what submission t wrote: throws where its call failed, so no result of a failed submission is read.
    virtual void raise(Ticket t) { (void)t; }
    // Whether submission t has retired, asked without waiting where the backend can tell; any thread may ask, as it may wait.
    virtual bool done(Ticket t) noexcept {
        wait(t);
        return true;
    }
    // Also retires work behind no ticket, including on failure paths.
    virtual void sync() noexcept = 0;

    // Copy to host memory after retiring earlier ops, throwing a failed submission's failure first; logits can instead use host_visible storage after a wait and raise.
    virtual void read(const Buffer& src, size_t off, void* dst, size_t bytes) = 0;

    // Storage to storage, within this backend. The KV cache grows with it.
    virtual void copy(Buffer& dst, size_t dst_off,
                      const Buffer& src, size_t src_off, size_t bytes) = 0;

    // Enqueue a host-to-storage copy, consuming the source before returning so callers can reuse staging immediately.
    virtual void write(Buffer& dst, size_t off, const void* src, size_t bytes) = 0;

    // Y[b*nout + o] = dot(row_o, X + b*nin) for all b in [0,nbatch) and o in [0,nout); X and Y are row-major with nbatch rows.
    // `type` is the quant type the registry is keyed by (the GGUF id), so every block format gets the batched path.
    virtual void matmul(WeightView weight, CSlice X,
                        Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // The output head: its results are the logits a caller reads directly, so a backend may keep more precise activations for it than for the projections inside the layers.
    virtual void matmul_logits(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch,
                               RowRuns runs = {}, Dtype dtype = Dtype::f16) {
        matmul(weight, X, Y, nin, nout, nbatch, runs, dtype);
    }

    // Y += W X, the projection whose output joins the residual stream: the model asks for the sum and each backend produces it its own way.
    // The CPU computes the product into scratch and adds; a device folds the add into the matmul's store, one dispatch fewer per projection.
    virtual void matmul_add(WeightView weight, CSlice X,
                            Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // Gather `count` rows of an embedding table into `dst`, row-major, `nin` floats each.
    // This is an op rather than a model-side read because the table is a Buffer: a device backend holds it in its own memory and the model cannot address it.
    virtual void embed(Slice dst, WeightView weight,
                       size_t nin, size_t nrows, const uint32_t* ids,
                       size_t count) = 0;

    // embed with the ids read on the device, `count` 32-bit integers in float-sized slots at `ids`, as a draft chain feeds a drafted token back (docs/SPECULATIVE.md, section 7): an id of `nrows` or more writes a zero row.
    virtual void embed_ids(Slice dst, WeightView weight, size_t nin, size_t nrows, CSlice ids, size_t count) {
        (void)dst; (void)weight; (void)nin; (void)nrows; (void)ids; (void)count;
        throw std::runtime_error(std::string("backend: ") + op_name(Op::embed_ids) + " is not implemented");
    }

    // For each of `rows` rows of `n` logits, the id of the largest, ties to the lowest, as a 32-bit integer in a float-sized slot of `ids`.
    // It is `n` where the largest is not finite or a logit is NaN, and for a row whose id in `after`, when given, is `n` or more, so a chain of drafts ends at its first invalid one.
    virtual void argmax_rows(Slice ids, CSlice logits, size_t rows, size_t n, CSlice after = {}) {
        (void)ids; (void)logits; (void)rows; (void)n; (void)after;
        throw std::runtime_error(std::string("backend: ") + op_name(Op::argmax_rows) + " is not implemented");
    }

    // Independent projections of the same X; outputs must not overlap each other, X, or any weights.
    // Outputs are observable after wait(), sync() or read(), as for matmul.
    virtual void matmul_group(std::initializer_list<Projection> projections,
                              CSlice X, size_t nin, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) {
        for (const auto& p : projections) {
            if (!p.weight.data.buffer) throw std::runtime_error("backend: projection without storage");
            matmul(p.weight, X, p.out, nin, p.rows, nbatch, runs, dtype);
        }
    }

    virtual KVLayout kv_layout() const = 0;

    // Allocates block-backed K/V up to max_tokens. Optional auxiliary widths specify F32 values per token for each layer, and pools the groups whose pooled rows each layer's blocks also keep (BlockKVStorage::auxiliary_pool).
    virtual std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv,
                                                size_t head_dim, size_t max_tokens,
                                                KVType k_type = KVType::f32,
                                                KVType v_type = KVType::f32,
                                                const std::vector<size_t>& auxiliary_widths = {},
                                                const std::vector<size_t>& auxiliary_pools = {}) = 0;

    // Store token-major [rows, n_head_kv, head_dim] rows, laid out in view order: view v owns the next views[v].nq rows, which go to positions length to length + nq of its sequence, and several views carry several sequences in one call.
    virtual void kv_write(size_t layer, const KVView* views, size_t n_views,
                          CSlice k, CSlice v, CSlice auxiliary = {}) = 0;

    // Causal GQA: Q/out are [rows, n_head, head_dim] in the same view order, and row b of view v attends through views[v].length + b.
    // A device backend gets every sequence of the pass in one launch.
    virtual void attention(CSlice Q, size_t layer, const KVView* views,
                           size_t n_views, Slice out,
                           int n_head, int n_head_kv, int head_dim) = 0;

    struct IndexArgs {
        size_t heads, dim, rope_dim, compression, budget;
        float eps;
    };
    // Q is already normalized and rotated; the groups each view's rows complete are pooled first (pool_kv), and every complete group's pooled key is read from the auxiliary blocks.
    // Output is U32 token IDs in chronological order, padded with UINT32_MAX to budget + compression - 1 per row; equal scores prefer the earlier group.
    virtual void select_kv(CSlice, size_t, const KVView*, size_t, Slice,
                           const IndexArgs&, CSlice, CSlice, CSlice, const GroupPositions& = {}) {
        throw std::runtime_error("backend: select_kv is not implemented");
    }
    // The complete groups the views' rows complete, their raw auxiliary keys averaged, normalized and rotated at the group's first position, into their pooled rows of the auxiliary blocks (BlockKVStorage::auxiliary_pool).
    // select_kv pools its own rows' groups; a caller that writes a pooled layer's auxiliary rows without selecting them pools them here, so every complete group of a history has its row.
    virtual void pool_kv(size_t, const KVView*, size_t, const IndexArgs&, CSlice, CSlice, CSlice, const GroupPositions& = {}) {
        throw std::runtime_error("backend: pool_kv is not implemented");
    }

    // Causal GQA over chronological unique U32 IDs, id_stride slots per query in view order; padding and IDs outside the causal prefix are ignored, and an empty selection yields zeros.
    virtual void sparse_attention(CSlice, size_t, const KVView*, size_t, CSlice, size_t, Slice, int, int, int) {
        throw std::runtime_error("backend: sparse_attention is not implemented");
    }

    // dst[i] = src[i] * rsqrt(mean(src^2) + eps) * w[i]  (RMS norm), the one-row case of rms_norm_rows.
    void rms_norm(Slice dst, CSlice src, CSlice w, size_t n, float eps) {
        rms_norm_rows(dst, src, w, 1, n, n, eps);
    }

    // The batched forms below exist so the model layer holds no elementwise loops and needs no host parallelism of its own.
    // Each is one call per layer instead of one per row (or per head, per row), which is what makes the graph expressible on a device: see docs/ROADMAP.md #4a.

    // RMS norm of `rows` rows of `n` floats against a shared weight, row r at src/dst + r*stride, src and dst aliasing only if identical.
    // `runs`, when given, are those of the matmul that reads dst next, so a device can write dst's activations in the form that matmul's kernel reads.
    virtual void rms_norm_rows(Slice dst, CSlice src, CSlice w,
                               size_t rows, size_t n, size_t stride, float eps, RowRuns runs = {}) = 0;

    // Per-head RMS norm over `head_dim` floats, then RoPE on the first `rope_dim` of them only, pairs (i, i + rope_dim / 2) at pos[r] in `cos`/`sin` tables of rope_dim / 2 entries a position, without the norm where `w` has no buffer.
    // Head h of row r starts at src + r * src_stride + h * src_head_stride, dst may alias src only if identical and contiguous, and `streams` give a row several positions (docs/src/backends-backend.md).
    virtual void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride,
                                   size_t heads, size_t head_dim, size_t rope_dim, CSlice w, float eps,
                                   CSlice cos, CSlice sin, const uint32_t* pos, RopeStreams streams = {}) = 0;

    // Normalize and rotate q and k in place over the whole head, then write k and v into the views' KV blocks using the primitive ops' row layouts.
    // A device may fuse these consecutive ops; k must still hold its normalized, rotated rows afterward.
    struct RopeArgs {
        CSlice cos, sin;
        size_t half;
        const uint32_t* pos;
        float eps;
    };
    virtual void norm_rope_kv(Slice q, size_t q_stride, size_t n_head, CSlice q_w,
                              Slice k, CSlice v, size_t kv_stride, size_t n_head_kv, CSlice k_w,
                              const RopeArgs& rope, size_t rows, size_t layer,
                              const KVView* views, size_t n_views) {
        const size_t dim = 2 * rope.half;
        norm_rope_partial(q, q, rows, q_stride, dim, n_head, dim, dim, q_w, rope.eps, rope.cos, rope.sin, rope.pos);
        norm_rope_partial(k, k, rows, kv_stride, dim, n_head_kv, dim, dim, k_w, rope.eps, rope.cos, rope.sin, rope.pos);
        kv_write(layer, views, n_views, k, v);
    }

    // The vision encoder's ops (docs/VISION.md, section 3), which a backend declares through implements.
    // LayerNorm of `rows` rows of `n` floats, row r at src/dst + r*stride: the mean and biased variance of the row, then (x - mean) / sqrt(var + eps) * w + b.
    virtual void layer_norm_rows(Slice dst, CSlice src, CSlice w, CSlice b, size_t rows, size_t n, size_t stride, float eps) {
        (void)dst; (void)src; (void)w; (void)b; (void)rows; (void)n; (void)stride; (void)eps;
        throw std::runtime_error(std::string("backend: ") + op_name(Op::layer_norm) + " is not implemented");
    }
    // dst[i] = gelu(src[i]) in the given form.
    virtual void gelu(Slice dst, CSlice src, size_t n, GeluForm form) {
        (void)dst; (void)src; (void)n; (void)form;
        throw std::runtime_error(std::string("backend: ") + op_name(Op::gelu) + " is not implemented");
    }
    // dst row r += bias, `rows` rows of `width` floats.
    virtual void add_bias(Slice dst, CSlice bias, size_t rows, size_t width) {
        (void)dst; (void)bias; (void)rows; (void)width;
        throw std::runtime_error(std::string("backend: ") + op_name(Op::add_bias) + " is not implemented");
    }
    // dst row r += the sum over k of weight[r*taps + k] times table row idx[r*taps + k], the taps taken in order; a row past `table_rows` is refused.
    virtual void mix_rows(Slice dst, CSlice table, size_t width, size_t table_rows, const uint32_t* idx, const float* weight, size_t rows, size_t taps) {
        (void)dst; (void)table; (void)width; (void)table_rows; (void)idx; (void)weight; (void)rows; (void)taps;
        throw std::runtime_error(std::string("backend: ") + op_name(Op::mix_rows) + " is not implemented");
    }

    // dst[i] = silu(gate[i]) * up[i], the SwiGLU elementwise stage.
    // `runs`, when given, group dst's rows of n / rows floats, `rows` being the last run's end, by prompt as matmul's runs do, for the matmul that reads dst next, as for rms_norm_rows.
    virtual void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, RowRuns runs = {}) = 0;

    // dst[i] += src[i], the residual add.
    virtual void add(Slice dst, CSlice src, size_t n) = 0;

    // dst row i = src row rows[i], `width` floats each.
    // Compacts the rows of a pass that want logits, which a batch mixing prefill and decode entries leaves non-contiguous, so the output head runs once over exactly those rows.
    virtual void gather_rows(Slice dst, CSlice src, size_t width,
                             const uint32_t* rows, size_t count) = 0;

    // Mixture of experts (docs/EXECUTION.md): a layer's router scores pick `k` of `n_expert` experts per token row, slot j of row r being entry r*k + j.
    // `ids` holds the chosen experts as 32-bit integers in float-sized slots, so they live in the activation arena and never leave the device.
    struct Routing {
        CSlice ids, weights;
        size_t k, n_expert;
    };

    // The id of a pick where a row has no candidate: its probabilities are not finite, from a NaN or infinite score or every score -inf.
    static constexpr uint32_t kNoExpert = 0xffffffffu;
    // A probability below FLT_MIN counts as 0, so such picks go by lower id whatever a device's denormal mode; HF orders them by value, which changes no output, since their weights are below 1e-38.

    // For each of `rows` rows of `scores` (n_expert floats each), the k experts of highest softmax probability over all n_expert, the most probable first and ties to the lower id, with their probabilities, divided by their sum when `normalize`.
    // A row without a candidate gives k picks of kNoExpert with weights 0.
    virtual void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize,
                               Slice ids, Slice weights) = 0;

    // Routed projections of one X: entry e = r*k + j of projection p is out[e*rows_p + o] = dot(row o of expert ids[e], X row r), for up to three projections, an entry of kNoExpert left unwritten.
    // A projection's data holds its n_expert matrices of `rows` rows each back to back; `runs` are matmul's, over the `nrows` token rows.
    virtual void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin,
                                size_t nrows, const Routing& routing, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // The routed projection whose output joins the residual stream: Y row r += sum over j in order of weights[e] * dot(expert ids[e], X row e), e = r*k + j, an entry of kNoExpert adding nothing.
    // The weighted sum is formed first and then added, so a row's result does not depend on how its slots were computed.
    virtual void matmul_experts_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout,
                                    size_t nrows, const Routing& routing, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // A routed feed-forward step over the normed rows h (MoeStep): routing, the experts' SwiGLU and weighted down sum, then a shared expert's SwiGLU under its row gate, joining y.
    // The base runs those ops in that order; a device may fuse them where it gives the same bits.
    virtual void moe_step(const MoeStep& m, CSlice h, Slice y, size_t nin, size_t rows, RowRuns runs = {}, Dtype dtype = Dtype::f16) {
        if (m.zero) repeat_row(y, *m.zero, rows, nin);
        const size_t n_expert = m.router.rows, ff = m.gate.rows;
        matmul(m.router.weight, h, m.router.out, nin, n_expert, rows, runs, Dtype::f32);
        route_experts(m.router.out, rows, n_expert, m.k, m.normalize, m.ids, m.weights);
        const Routing routing{m.ids, m.weights, m.k, n_expert};
        matmul_experts({m.gate, m.up}, h, nin, rows, routing, runs, dtype);
        // The routed down projection reads the SiLU's output as k entries a token row, each of its token's prompt.
        std::vector<RowRun>& entry_runs = *m.scratch;
        entry_runs.clear();
        for (size_t i = 0; i < runs.n; ++i) entry_runs.push_back(RowRun{runs.runs[i].end * m.k, runs.runs[i].extent});
        silu_mul(m.act, m.gate.out, m.up.out, rows * m.k * ff, {entry_runs.data(), entry_runs.size()});
        matmul_experts_add(m.down, m.act, y, ff, nin, rows, routing, runs, dtype);
        if (!m.shared) return;
        const MoeShared& s = *m.shared;
        const bool gated = s.row_gate.weight.data.buffer != nullptr;
        if (gated) matmul(s.row_gate.weight, h, s.row_gate.out, nin, 1, rows, runs, dtype);
        matmul_group({s.gate, s.up}, h, nin, rows, runs, dtype);
        silu_mul(s.act, s.gate.out, s.up.out, rows * s.gate.rows, runs);
        if (gated) sigmoid_mul(s.act, s.act, s.row_gate.out, rows, s.gate.rows, 1, 1, 0, runs);
        matmul_add(s.down, s.act, y, s.gate.rows, nin, rows, runs, dtype);
    }

    // Copy the first `width` floats of `row` into each of `rows` rows of dst, then double the copied rows, so clearing needs no host buffer or per-token upload.
    void repeat_row(Slice dst, const Buffer& row, size_t rows, size_t width) {
        if (!rows) return;
        copy(*dst.buffer, dst.offset * sizeof(float), row, 0, width * sizeof(float));
        for (size_t done = 1; done < rows;) {
            const size_t count = std::min(done, rows - done);
            copy(*dst.buffer, (dst.offset + done * width) * sizeof(float), *dst.buffer, dst.offset * sizeof(float), count * width * sizeof(float));
            done += count;
        }
    }

    // The ops of the qwen35 layers (docs/QWEN35.md, The forward pass), with norm_rope_partial above.

    // The recurrent state of `layers` linear-attention layers with `slots` slots each, allocated now and zero-filled, so no pass allocates state.
    std::unique_ptr<StateStorage> state_alloc(size_t layers, size_t slots, const StateShape& shape,
                                             const std::vector<size_t>& tail_floats = {}) {
        if (!layers || !slots || !shape.k_heads || !shape.k_dim || !shape.v_dim || !shape.v_heads || shape.v_heads % shape.k_heads)
            throw std::runtime_error("backend: invalid state shape");
        if (!tail_floats.empty() && tail_floats.size() != layers)
            throw std::runtime_error("backend: state tail count differs from its layers");
        std::vector<size_t> bytes(layers);
        for (size_t l = 0; l < layers; ++l) {
            const size_t stride = size_add(shape.slot_floats(), tail_floats.empty() ? 0 : tail_floats[l]);
            bytes[l] = size_mul(size_mul(slots, stride), sizeof(float));
        }
        std::vector<BufferPtr> buffers;
        buffers.reserve(layers);
        for (size_t l = 0; l < layers; ++l) buffers.push_back(alloc(bytes[l]));
        return std::make_unique<StateStorage>(std::move(buffers), slots, shape, tail_floats);
    }

    // The linear-attention layers' causal conv, then SiLU: out[t][c] = silu(sum over tap i of w[c * kConvTaps + i] * x[t - kConvTaps + 1 + i][c]), w being `ssm_conv1d` as stored, so tap kConvTaps - 1 multiplies row t.
    // x and out are the views' rows of the storage's channels(), in view order; a view reads the rows before its first from slot src, rows before its sequence's start being zero, and leaves its last kConvTaps - 1 raw rows in slot dst.
    virtual void causal_conv_silu(Slice out, CSlice x, CSlice w, size_t layer, const StateView* views, size_t n_views) = 0;

    // The gated delta rule of the linear-attention layers (docs/QWEN35.md, Linear attention, steps 3 to 5), token by token for every (view, V head) from slot src's matrices into slot dst's.
    // qkv holds the conv's output rows [q | k | v] of channels(); q and k are L2-normed with kL2NormEps, q scaled by 1 / sqrt(k_dim), beta is sigmoid(b), and the decay exp(a * softplus(alpha + dt_bias)) is 0 below 2^-126.
    virtual void gated_delta_rule(Slice out, CSlice qkv, CSlice alpha, CSlice b, CSlice a, CSlice dt_bias,
                                  size_t layer, const StateView* views, size_t n_views) = 0;

    // dst = RMSNorm(x; w) times the selected gate of z over each head; w is one dim-wide weight shared by every head.
    // dst may alias x only if identical, and never z; `runs` as for rms_norm_rows.
    virtual void gated_rms_norm(Slice dst, CSlice x, CSlice z, CSlice w, size_t rows, size_t heads, size_t dim, float eps,
                                RowRuns runs = {}, NormGate gate = NormGate::silu) = 0;

    // dst[r][h][d] = x[r][h][d] * sigmoid(gate[r * gate_stride + h * gate_head_stride + d]), x and dst being `rows` rows of heads * dim floats.
    // The output gate reads each query head's gate in place from `attn_q`'s rows, a scale of one value per row is heads = the row's width, dim = 1 and gate_head_stride = 0, and dst may alias x only if identical; `runs` as for silu_mul.
    virtual void sigmoid_mul(Slice dst, CSlice x, CSlice gate, size_t rows, size_t heads, size_t dim,
                             size_t gate_stride, size_t gate_head_stride, RowRuns runs = {}) = 0;

    // Normalize each dim-wide group separately with its own weight slice in each row of groups * dim floats; dst may equal src.
    virtual void group_rms_norm(Slice dst, CSlice src, CSlice w, size_t rows, size_t groups, size_t dim, float eps,
                                RowRuns runs = {}) = 0;

    // dst = silu(src * scale) for rows of width floats; dst may equal src.
    virtual void silu_scaled(Slice dst, CSlice src, size_t rows, size_t width, float scale, RowRuns runs = {}) = 0;

    // Reduce each row's streams * dim normalized values with elementwise sigmoid(gate), dividing the sum across streams by streams into dim outputs.
    virtual void mix_streams(Slice dst, CSlice src, CSlice gate, size_t rows, size_t streams, size_t dim,
                             RowRuns runs = {}) = 0;

    // Add each dim-wide block row to every residual stream, weighted by 2 * sigmoid(gate[row][stream] / streams); dst holds the residual in place.
    virtual void inject_streams(Slice dst, CSlice block, CSlice gate, size_t rows, size_t streams, size_t dim) = 0;

    // For each normalized key/query group, g = dot(key, query) / sqrt(dim), then sigmoid(sign(g) * sqrt(max(abs(g), 1e-6))) gates the row's shared dim-wide value.
    virtual void signed_sqrt_gate(Slice dst, CSlice key, CSlice query, CSlice value, size_t rows, size_t groups, size_t dim,
                                  RowRuns runs = {}) = 0;

    // Add SiLU of a causal depthwise convolution of x to residual, using weights [channel][tap], last tap current, and taps spaced by dilation.
    // Carries (taps - 1) * dilation raw x rows in the layer's F32 tail after its common state prefix; exact out=residual is allowed, but out may not overlap x or weights.
    virtual void dilated_conv_silu_add(Slice out, CSlice x, CSlice residual, CSlice w, size_t channels, size_t taps, size_t dilation,
                                       size_t layer, const StateView* views, size_t n_views) = 0;
protected:
    void record_matrix_path(MatrixPath path) { matrix_paths_.record(path); }

private:
    MatrixPaths matrix_paths_;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
