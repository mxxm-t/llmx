#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/bf16.hpp"
#include "core/fp16.hpp"
#include "core/host_memory.hpp"
#include "core/job_threads.hpp"
#include "format/file_reader.hpp"
#include "format/gguf.hpp"
#include "format/hf_quant.hpp"
#include "inference/read_ring.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/chat.hpp"
#include "inference/directory.hpp"
#include "inference/pair.hpp"
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/shard.hpp"
#include "model/arch/registry.hpp"

// Loading a model file, the one sequence every command and tool opens a model through (docs/src/inference-load.md).

namespace infer {

// How a load reads the weights, the CLI's --load-mode, the same on every backend; the first is the default.
// `automatic` streams what a copying backend takes and maps files only for weights a host reads in place, `mapped` maps and copies as the model is built, and `direct` streams every weight around the file cache and maps nothing.
enum class LoadMode { automatic, mapped, direct };

inline LoadMode load_mode_of(const std::string& name) {
    if (name == "auto") return LoadMode::automatic;
    if (name == "mapped") return LoadMode::mapped;
    if (name == "direct") return LoadMode::direct;
    throw std::runtime_error("unknown load mode '" + name + "' (auto, mapped or direct)");
}
inline const char* load_mode_name(LoadMode m) { return m == LoadMode::mapped ? "mapped" : m == LoadMode::direct ? "direct" : "auto"; }

// What a load is doing, for its caller to show: its phase (`metadata`, `fit`, `allocate`, `weights`) and in `weights` the bytes read, of every file and of `file` (docs/src/inference-load.md).
struct LoadStep {
    LoadStep(const char* p, size_t c = 0, size_t t = 0, std::string f = {}, size_t fc = 0, size_t ft = 0)
        : phase(p), completed(c), total(t), file(std::move(f)), file_completed(fc), file_total(ft) {}
    const char* phase;
    size_t completed, total;
    std::string file;
    size_t file_completed, file_total;
};
using LoadReport = std::function<void(const LoadStep&)>;

// Where a load's time went, in seconds, for the CLI's timing line: building the model, and for a streamed load the busiest reader thread's reads, the uploads, and the uploads' waits for a read.
struct LoadTimes {
    LoadMode mode{};
    size_t files = 0;           // files the weights were streamed from through the file cache
    size_t direct_files = 0;    // files they were streamed from around it
    size_t streamed = 0;        // bytes read for them, gaps and granule rounding included
    size_t copied = 0;          // bytes the devices copied straight out of the reads, each device's counted, where the rest go through their staging
    double construct = 0, read = 0, upload = 0, wait = 0;
};

// A model file loaded for use: the file, its tokenizer and chat format, and the model placed over the caller's backends.
// A host backend's buffers read the file's payload, or a direct load's copy of it, in place while the model lives, so load_model builds this in place and it is never copied or moved; `model` is declared last, so it is destroyed first.
struct LoadedModel {
    gguf::GGUFModel file;
    std::optional<bpe::Tokenizer> tok;
    chat::ChatFormat chat;
    bool replaced_template = false;  // ModelOptions::chat_template took the place of a template the file carried
    std::string plan;               // what each device of a split was given (LayerSplit::describe), empty otherwise
    DtypePlan dtype;
    size_t checkpoint_kv_tokens = 0; // the KV tokens a server's fitted checkpoint slots took from its budget (PlacedModel)
    std::string kv_kept_context;     // why a server's automatic budget kept the model's context (PlacedModel)
    std::optional<spec::DrafterKind> drafter;   // what the drafter file beside the model is (spec::pair), when one was given
    std::vector<backend::BackendPtr> backends;  // the backends the caller gave, which a draft model beside the model shares
    LoadTimes times;
    std::vector<std::shared_ptr<core::HostPages>> host;   // direct copies, shared with any immutable table that reads them
    std::vector<float> widened;     // a directory's 16-bit vectors as F32 (inference/directory.hpp), which a host reads in place
    std::map<size_t, core::HostPages> made;   // a directory's repacked tensors a host reads in place (format/hf_quant.hpp), by tensor, each at least tensor_bytes long
    std::unique_ptr<Model> model;

    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
};

// A weight a copying backend took: the storage the model's construction allocated for it (Backend::alloc_weight), filled from the file once the model is built.
// A tensor group's member takes only its shard, the copy `shard` describes (model/shard.hpp); without one it takes the whole tensor.
struct Upload {
    size_t tensor = 0;
    backend::Backend* backend = nullptr;
    backend::BufferPtr buffer;
    std::optional<shard::Held> shard;
};

// Puts the bytes [lo, lo + n) of a tensor at `data` that `held` names in a member's storage: runs under a page packed and written once while they follow one another there, longer ones by `send`.
template <class Write, class Send>
inline void put_runs(const shard::Held& held, size_t lo, size_t n, const uint8_t* data, std::vector<uint8_t>& packed, Write write, Send send) {
    static constexpr size_t kPackedRun = 4096;
    size_t packed_to = 0;
    auto flush = [&] {
        if (!packed.empty()) write(packed_to, packed.data(), packed.size());
        packed.clear();
    };
    shard::file_runs(held, lo, lo + n, [&](const shard::Run& r) {
        if (r.bytes >= kPackedRun) {
            flush();
            send(r.from, r.to, r.bytes);
            return;
        }
        if (!packed.empty() && r.to != packed_to + packed.size()) flush();
        if (packed.empty()) packed_to = r.to;
        packed.insert(packed.end(), data + (r.from - lo), data + (r.from - lo) + r.bytes);
    });
    flush();
}

// What the loader's hook records while the model is built: per tensor whether a backend that reads in place took it, and every weight a copying backend took.
struct WeightPlan {
    std::vector<char> host_reads;
    std::vector<Upload> uploads;
    std::vector<char> resident;   // per tensor, whether its bytes are the loader's own host data rather than the file's, adopted at once and never streamed; empty for none
    std::vector<char> composite;  // per tensor, whether its bytes are made at load (format/hf_quant.hpp) rather than read; empty for none
    std::vector<char> made_host;  // per composite, whether a host took the bytes made for it, which then stay
};

// load_model's adoption hook: a backend that reads in place adopts a weight where it lies, and one that copies adopts it at once unless `defer` gives a weight not plan.resident storage to stream into (plan.uploads).
// The model hands each tensor to each of `backends` about once whole and once as a shard, which sizes the records; a record that cannot be kept drains the backend before its storage goes, as a failed write does.
inline AdoptWeight planning_adopt(const ModelWeights& weights, size_t backends, WeightPlan& plan, bool defer, std::function<void(size_t, bool)> make,
                                  std::function<void(size_t)> release) {
    plan.host_reads.assign(weights.tensors.size(), 0);
    plan.uploads.clear();
    plan.uploads.reserve(defer ? 2 * weights.tensors.size() * backends : 0);
    return [&weights, &plan, defer, make, release](size_t i, backend::Backend& b, const shard::Held* held) {
        const TensorView& t = weights.tensors[i];
        const bool resident = i < plan.resident.size() && plan.resident[i], later = defer && !resident;
        const bool composite = i < plan.composite.size() && plan.composite[i];
        // make(i, now) points the view at composite i's storage, filled at once when `now`; a host does not read a weight inside adopt, so its fill waits for the build.
        const bool host = !held && b.reads_in_place();
        if (composite && (host || !later)) make(i, !host);
        if (composite && host) plan.made_host[i] = 1;
        const auto copied = [&](backend::BufferPtr buffer) {
            if (composite && !plan.made_host[i]) release(i);
            return buffer;
        };
        const auto record = [&](const backend::BufferPtr& buffer, std::optional<shard::Held> part) {
            try {
                plan.uploads.push_back({i, &b, buffer, std::move(part)});
            } catch (...) {
                b.sync();
                throw;
            }
        };
        if (held) {
            backend::BufferPtr buffer = b.alloc_weight(held->bytes);
            // A shard's zero runs are written here, in either load, so no storage is trusted to hold zeros; the file's runs are streamed or written from the mapping.
            // Until it returns, the storage is this call's alone: a write that fails leaves the writes before it queued on it, so the backend drains before the storage goes.
            std::vector<uint8_t> zeros, packed;
            try {
                shard::zero_runs(*held, [&](const shard::Run& r) {
                    zeros.resize(std::max(zeros.size(), r.bytes));
                    b.write(*buffer, r.to, zeros.data(), r.bytes);
                });
                if (!later) {
                    const auto write = [&](size_t to, const uint8_t* from, size_t n) { b.write(*buffer, to, from, n); };
                    put_runs(*held, 0, t.bytes, t.data, packed, write, [&](size_t from, size_t to, size_t n) { b.write(*buffer, to, t.data + from, n); });
                }
            } catch (...) {
                b.sync();
                throw;
            }
            if (later) record(buffer, *held);
            return later ? buffer : copied(std::move(buffer));
        }
        if (b.reads_in_place()) {
            plan.host_reads[i] = !resident && !composite;
            return b.adopt(t.data, t.bytes);
        }
        if (!later) return copied(b.adopt(t.data, t.bytes));
        backend::BufferPtr buffer = b.alloc_weight(t.bytes);
        record(buffer, std::nullopt);
        return buffer;
    };
}

namespace detail {

inline double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

inline std::string gib(size_t bytes) {
    char text[32];
    std::snprintf(text, sizeof text, "%.2f GiB", double(bytes) / double(size_t(1) << 30));
    return text;
}

// Read the pages of `tensors` in, reporting their bytes to `progress`.
// Tensors larger than the memory available are left to be read once by their reader, and the progress goes straight to complete.
inline void warm(const gguf::GGUFModel& file, const std::vector<size_t>& tensors, const format::LoadProgress& progress) {
    const size_t bytes = gguf::bytes_of(file, tensors);
    const auto available = core::host_memory_available();
    if (!available || bytes <= *available) return gguf::warm(file, tensors, progress);
    if (progress) {
        progress(0, bytes);
        progress(bytes, bytes);
    }
}

// Whether a streamed load reads around the file cache when it may: when the bytes it streams are more than the host's available memory, which would evict them before they are uploaded.
inline bool reads_around(size_t streamed, std::optional<size_t> available) {
    return available && streamed > *available;
}

// A reader for the file at `path`, direct when `around` asks and the file system takes direct reads, and through the cache otherwise.
inline std::unique_ptr<format::FileReader> open_reader(const std::string& path, bool around) {
    if (around) {
        try {
            return std::make_unique<format::FileReader>(path, true);
        } catch (const format::DirectUnavailable&) {
        }
    }
    return std::make_unique<format::FileReader>(path);
}

// One read of the stream: `bytes` from `offset` in file `file`, both on the file's granule, and the parts of tensors it holds.
// A piece with `into` is read straight there, into a direct load's copy of the file the host reads in place, and nothing is written from it; one with `make` is one composite tensor, which `make` writes into its slot on a reader thread.
struct Piece {
    struct Part {
        size_t tensor = 0, tensor_offset = 0, piece_offset = 0, bytes = 0;
    };
    size_t file = 0;
    uint64_t offset = 0;
    size_t bytes = 0;
    std::vector<Part> parts;
    uint8_t* into = nullptr;
    std::function<void(uint8_t*)> make{};
};

// Composite `i`'s bytes made at `out` from its parts by hf_quant::repack.
// Its parts come from the mapping when `mapping` and their file is mapped, else each is read whole through `reader_of` on its granule into one buffer, and as it holds no state several threads run it at once.
inline void make_composite(const gguf::GGUFModel& file, size_t i, const std::function<const format::FileReader&(const std::string&)>& reader_of, uint8_t* out,
                           bool mapping) {
    const gguf::GGUFModel::Composite& c = file.composites.at(i);
    struct Read {
        const format::FileReader* reader;
        uint64_t from;
        size_t lead, bytes, request, at, part;
    };
    std::vector<Read> reads;
    std::vector<const uint8_t*> parts(c.parts.size(), nullptr);
    size_t total = 0;
    for (size_t k = 0; k < c.parts.size(); ++k) {
        const size_t offset = c.parts[k].second, bytes = size_t(c.parts[k].first.data_size());
        if (mapping && (parts[k] = file.part_data(offset, bytes))) continue;
        const format::FileSpan span = file.part_span(offset, bytes);
        const format::FileReader& reader = reader_of(span.file);
        const size_t g = reader.direct() ? reader.granule() : 1, lead = size_t(span.offset % g);
        const size_t request = (lead + span.bytes + g - 1) / g * g;
        reads.push_back({&reader, span.offset - lead, lead, span.bytes, request, total, k});
        total += (request + core::page_size() - 1) / core::page_size() * core::page_size();
    }
    core::HostPages buffer(total);
    for (const Read& r : reads) {
        if (r.reader->read(r.from, buffer.data() + r.at, r.request) < r.lead + r.bytes) throw std::runtime_error("loading: short file read");
        parts[r.part] = buffer.data() + r.at + r.lead;
    }
    hf_quant::repack(c, file.tensors[i], parts, out);
    if (mapping)
        for (const auto& part : c.parts) file.drop_part(part.second, size_t(part.first.data_size()));
}

// The reads that bring `tensors` (distinct, in file order, tensor t at spans[t] in file file_of[t]) into memory.
// A read starts on its file's granule and holds at most `limit` bytes, rounded down to whole granules, and carries on over the tensors that follow in the same file while no gap between them is longer than a granule.
inline std::vector<Piece> plan_pieces(const std::vector<size_t>& tensors, const std::vector<format::FileSpan>& spans,
                                      const std::vector<size_t>& file_of, const std::vector<size_t>& granules, size_t limit) {
    std::vector<Piece> pieces;
    const size_t none = SIZE_MAX;
    size_t open = none;
    for (size_t t : tensors) {
        const format::FileSpan& s = spans[t];
        const size_t f = file_of[t], g = granules[f], cap = std::max(g, limit / g * g);
        uint64_t pos = s.offset;
        for (size_t done = 0; done < s.bytes;) {
            if (open != none) {
                const Piece& p = pieces[open];
                if (p.file != f || pos < p.offset || pos > p.offset + p.bytes + g || pos >= p.offset + cap) open = none;
            }
            if (open == none) {
                pieces.push_back(Piece{f, pos / g * g, 0, {}});
                open = pieces.size() - 1;
            }
            Piece& p = pieces[open];
            const size_t at = size_t(pos - p.offset), n = std::min(s.bytes - done, cap - at);
            p.parts.push_back({t, done, at, n});
            p.bytes = std::max(p.bytes, at + n);
            pos += n;
            done += n;
        }
    }
    for (Piece& p : pieces) p.bytes = (p.bytes + granules[p.file] - 1) / granules[p.file] * granules[p.file];
    return pieces;
}

// The stream's reads in flight and its slots: eight 16 MiB reads keep a striped pool busy (docs/STATUS.md, Server startup profiled end to end).
inline constexpr size_t kStreamReaders = 8, kStreamSlots = 16;

// Read `pieces` through the read ring (ReadRing) and, on this thread, send each part to every upload of its tensor (`destinations`) in order, then report its file and bytes to `streamed`.
// A backend that reads a slot in place (Backend::wrap_host) takes a part as a device copy out of it and the slot goes back once the copies retire; whatever fails, the copies are retired and the readers joined before the ring is freed.
inline void stream(const std::vector<Piece>& pieces, const std::vector<std::shared_ptr<const format::FileReader>>& readers,
                   const std::vector<std::vector<const Upload*>>& destinations, const std::function<void(size_t, size_t)>& streamed,
                   LoadTimes& times) {
    if (pieces.empty()) return;
    size_t slot_bytes = 0;
    for (const Piece& p : pieces) slot_bytes = std::max(slot_bytes, p.bytes);
    ReadRing ring(pieces.size(), slot_bytes, [&](size_t k, uint8_t* into) {
        const Piece& p = pieces[k];
        if (p.make) {
            p.make(into);
            return p.bytes;
        }
        return readers[p.file]->read(p.offset, p.into ? p.into : into, p.bytes);
    }, kStreamSlots, kStreamReaders);
    const size_t slots = ring.slots();

    // The backends the parts go to, each with its view of every slot where it reads them in place, and per slot the ticket of its last copies out of it.
    // The views are declared after the ring, so they go before it.
    std::vector<backend::Backend*> targets;
    for (const auto& list : destinations)
        for (const Upload* u : list)
            if (std::find(targets.begin(), targets.end(), u->backend) == targets.end()) targets.push_back(u->backend);
    std::vector<std::vector<backend::BufferPtr>> views(targets.size());
    for (size_t t = 0; t < targets.size(); ++t)
        for (size_t i = 0; i < slots; ++i) {
            backend::BufferPtr v = targets[t]->wrap_host(ring.pages(i).data(), ring.pages(i).size());
            if (!v) {
                views[t].clear();
                break;
            }
            views[t].push_back(std::move(v));
        }
    ring.start();
    std::vector<std::vector<backend::Ticket>> copying(slots, std::vector<backend::Ticket>(targets.size(), 0));
    auto target_of = [&](const backend::Backend* b) { return size_t(std::find(targets.begin(), targets.end(), b) - targets.begin()); };
    // Retire the copies out of `slot`, after which the readers may refill it.
    auto retire = [&](size_t slot) {
        for (size_t t = 0; t < targets.size(); ++t)
            if (copying[slot][t]) {
                targets[t]->wait(copying[slot][t]);
                copying[slot][t] = 0;
            }
    };

    try {
        std::vector<uint8_t> packed;
        for (size_t k = 0; k < pieces.size(); ++k) {
            const auto w0 = std::chrono::steady_clock::now();
            const ReadRing::Full full = *ring.next(true);
            times.wait += seconds_since(w0);
            const size_t n = full.bytes, slot = full.slot;
            const Piece& p = pieces[k];
            const uint8_t* data = p.into ? p.into : ring.pages(slot).data();
            const auto t0 = std::chrono::steady_clock::now();
            size_t bytes = 0;
            for (const Piece::Part& part : p.parts) {
                if (part.piece_offset + part.bytes > n)
                    throw std::runtime_error(readers[p.file]->path() + " ended at " + std::to_string(p.offset + n) + " bytes, before its tensors");
                for (const Upload* u : destinations[part.tensor]) {
                    const size_t t = target_of(u->backend);
                    // `count` bytes of the part from `from` in the tensor, to `to` in the storage.
                    auto send = [&](size_t from, size_t to, size_t count) {
                        const size_t at = part.piece_offset + (from - part.tensor_offset);
                        if (!p.into && !views[t].empty()) {
                            u->backend->copy(*u->buffer, to, *views[t][slot], at, count);
                            copying[slot][t] = 1;
                            times.copied += count;
                        } else {
                            u->backend->write(*u->buffer, to, data + at, count);
                        }
                    };
                    if (!u->shard) {
                        send(part.tensor_offset, part.tensor_offset, part.bytes);
                        continue;
                    }
                    const auto write = [&](size_t to, const uint8_t* from, size_t n) { u->backend->write(*u->buffer, to, from, n); };
                    put_runs(*u->shard, part.tensor_offset, part.bytes, data + part.piece_offset, packed, write, send);
                }
                bytes += part.bytes;
            }
            for (size_t t = 0; t < targets.size(); ++t)
                if (copying[slot][t]) copying[slot][t] = targets[t]->submit();
            // The device copies this piece while this thread queues the next, so the slot of the piece before is the one waited for here.
            if (k) retire((k - 1) % slots);
            const bool held = std::any_of(copying[slot].begin(), copying[slot].end(), [](backend::Ticket x) { return x != 0; });
            times.upload += seconds_since(t0);
            times.streamed += p.bytes;
            if (k) ring.free(k - 1);
            if (!held) ring.free(k);
            if (streamed) streamed(p.file, bytes);
        }
        const auto t0 = std::chrono::steady_clock::now();
        retire((pieces.size() - 1) % slots);
        times.upload += seconds_since(t0);
    } catch (...) {
        ring.stop();
        for (backend::Backend* b : targets) b->sync();
        throw;
    }
    ring.stop();
    times.read = ring.reading();
}

// Finish a header-only load after its architecture has read the metadata.
inline std::unique_ptr<LoadedModel> load_weights(std::unique_ptr<LoadedModel> loaded, ModelWeights weights,
                                                std::vector<backend::BackendPtr> backends, const PlacementRequest& request,
                                                const ModelOptions& options, const LoadReport& report, LoadMode mode, bool widen) {
    gguf::GGUFModel& file = loaded->file;
    loaded->times.mode = mode;
    std::vector<size_t> every(file.tensors.size());
    for (size_t i = 0; i < every.size(); ++i) every[i] = i;
    const bool host = host_reads_in_place(backends, request);
    // The files, in the payload's order, and a reader for each once one is opened; a direct load opens them all first, so a file system that does not take direct reads is refused before anything is built.
    std::vector<std::string> paths;
    for (const auto& seg : file.segments)
        if (paths.empty() || paths.back() != seg.path) paths.push_back(seg.path);
    std::vector<std::shared_ptr<const format::FileReader>> readers(paths.size());
    auto file_index = [&](const std::string& p) { return size_t(std::find(paths.begin(), paths.end(), p) - paths.begin()); };
    // A file's reader, opened once, direct under direct and buffered otherwise, and held to the size its header gives.
    const auto reader_of = [&](size_t f) -> const format::FileReader& {
        if (!readers[f]) {
            readers[f] = std::make_shared<const format::FileReader>(paths[f], mode == LoadMode::direct);
            gguf::check_size(file, paths[f], readers[f]->size());
        }
        return *readers[f];
    };
    if (mode == LoadMode::direct) {
        for (size_t f = 0; f < paths.size(); ++f) {
            try {
                reader_of(f);
            } catch (const format::DirectUnavailable& e) {
                throw std::runtime_error(std::string("--load-mode direct: ") + e.what());
            }
        }
    }
    // A tensor's bytes read from its span into `to`, through a page-aligned window when its reader is direct.
    core::HostPages window;
    const auto read_span = [&](size_t i, uint8_t* to) {
        const format::FileSpan span = file.span(i);
        const format::FileReader& reader = reader_of(file_index(span.file));
        if (reader.direct()) {
            if (!window.data()) {
                window = core::HostPages::reserved(reader.granule());
                window.commit(0, reader.granule());
            }
            format::read_aligned(reader, span.offset, to, span.bytes, window.data(), reader.granule(), "loading");
        } else if (reader.read(span.offset, to, span.bytes) != span.bytes) {
            throw std::runtime_error("loading: short file read");
        }
    };
    // A directory's 16-bit vectors are widened to F32, exactly, into the loader's buffer, as the converter stores them; they are adopted at once and never streamed or warmed.
    std::vector<char> resident(file.tensors.size(), 0);
    if (widen) {
        size_t floats = 0;
        for (size_t i : every)
            if (widened_on_load(file.tensors[i])) floats += size_t(file.tensors[i].n_elements());
        loaded->widened.resize(floats);
        std::vector<uint16_t> half;
        size_t at = 0;
        for (size_t i : every) {
            const gguf::TensorInfo& info = file.tensors[i];
            if (!widened_on_load(info)) continue;
            const size_t n = size_t(info.n_elements());
            half.resize(n);
            read_span(i, reinterpret_cast<uint8_t*>(half.data()));
            float* out = loaded->widened.data() + at;
            for (size_t k = 0; k < n; ++k) out[k] = info.type == quant::GGML_TYPE_BF16 ? bf16_to_f32(half[k]) : f16_to_f32(half[k]);
            TensorView& t = weights.tensors[i];
            t.type = quant::GGML_TYPE_F32;
            t.data = reinterpret_cast<const uint8_t*>(out);
            t.bytes = n * sizeof(float);
            resident[i] = 1;
            at += n;
        }
    }
    // A directory's repacked tensors (format/hf_quant.hpp) are made as they are adopted when a copy wants them at once, after the build, side by side, when only a host reads them, and on the stream's readers otherwise.
    std::vector<char> composite(file.tensors.size(), 0);
    for (const auto& item : file.composites) composite[item.first] = 1;
    const auto reader_at = [&](const std::string& path) -> const format::FileReader& { return reader_of(file_index(path)); };
    std::vector<size_t> unfilled;   // composites a host took whose bytes are made after the build
    const auto fill = [&](size_t i) { make_composite(file, i, reader_at, loaded->made.at(i).data(), true); };
    const auto make = [&](size_t i, bool now) {
        const auto waiting = std::find(unfilled.begin(), unfilled.end(), i);
        if (!weights.tensors[i].data) {
            weights.tensors[i].data = loaded->made.insert_or_assign(i, core::HostPages(file.tensor_bytes(i))).first->second.data();
            if (!now) unfilled.push_back(i);
            else fill(i);
        } else if (now && waiting != unfilled.end()) {
            unfilled.erase(waiting);
            fill(i);
        }
    };
    const auto release = [&](size_t i) {
        weights.tensors[i].data = nullptr;
        loaded->made.erase(i);
    };
    std::vector<LookupCandidate> lookups;
    const ModelPlan architecture = plan_model(weights, request.drafter);
    for (const LayerPlan& layer : architecture.layers) {
        if (!layer.lookup) continue;
        const auto role = std::find_if(layer.roles.begin(), layer.roles.end(), [&](const Role& r) { return r.id == layer.lookup->role; });
        if (role == layer.roles.end() || !role->tensor) throw std::runtime_error("loading: missing lookup table role");
        const size_t i = *role->tensor;
        if (std::any_of(lookups.begin(), lookups.end(), [&](const LookupCandidate& c) { return c.tensor == i; })) continue;
        const TensorView& t = weights.tensors[i];
        if (t.shape.size() != 2) throw std::runtime_error("loading: lookup table is not a matrix: " + t.name);
        const format::FileSpan span = file.span(i);
        const size_t f = file_index(span.file);
        reader_of(f);
        lookups.push_back({i, std::make_shared<const WeightRows>(t.type, size_t(t.shape[0]), size_t(t.shape[1]), readers[f], span.offset, span.bytes)});
    }
    std::optional<PlannedModel> placement;
    if (!lookups.empty()) {
        if (report) report({"fit"});
        placement = plan_placement(weights, backends, request, options, lookups);
    }
    // A table read from its file reads through its own reader, so the payload is mapped only for weights a host reads in place; the table never joins the stream or the warming.
    if (mode == LoadMode::mapped || (mode == LoadMode::automatic && host)) gguf::map_payload(file);
    for (size_t i : every)
        if (!resident[i]) weights.tensors[i].data = file.tensor_data(i);
    // A direct load gives a host a copy of each file laid out as the file, so every weight keeps the offset within its page the mapping would give it: address space now, memory once the model says which weights the host reads.
    if (mode == LoadMode::direct && host) {
        // Each copy covers its file as its header gave it, rounded up to the reads' granule, since the last read ends on it.
        for (size_t f = 0; f < paths.size(); ++f) {
            const uint64_t g = readers[f]->granule();
            loaded->host.push_back(std::make_shared<core::HostPages>(core::HostPages::reserved(size_t(gguf::aligned_size(gguf::file_size(file, paths[f]), g)))));
        }
        for (size_t i : every) {
            if (resident[i] || composite[i]) continue;
            const format::FileSpan s = file.span(i);
            weights.tensors[i].data = loaded->host[file_index(s.file)]->data() + s.offset;
        }
    }
    std::vector<char> external = resident;
    for (size_t i : every) external[i] |= composite[i];
    for (size_t n = 0; n < lookups.size(); ++n) {
        const size_t i = lookups[n].tensor;
        TensorView& t = weights.tensors[i];
        if (placement->lookups[n] == LookupStorage::device) continue;
        external[i] = 1;
        // In every mode the rows are read through the table's readers from the file's reader, direct under direct and buffered otherwise, never from a mapping.
        t.row_source = lookups[n].file;
        t.data = nullptr;
    }
    // Source roles' bytes, read from their spans in every mode for fill_tables and never placed; they go when the load ends (docs/src/inference-load.md).
    std::vector<std::vector<uint8_t>> sources;
    std::vector<size_t> source_tensors;
    for (const LayerPlan& layer : architecture.layers)
        for (const Role& role : layer.roles) {
            if (role.kind != RoleKind::source || !role.tensor) continue;
            const size_t i = *role.tensor;
            if (resident[i]) throw std::runtime_error("loading: a widened vector is a source role: " + file.tensors[i].name);
            if (composite[i]) throw std::runtime_error("loading: a repacked tensor is a source role: " + file.tensors[i].name);
            if (external[i]) continue;
            sources.emplace_back(file.span(i).bytes);
            read_span(i, sources.back().data());
            weights.tensors[i].data = sources.back().data();
            external[i] = 1;
            source_tensors.push_back(i);
        }
    // The bytes of the weights read, of every file, for the report.
    const format::LoadProgress progress = [&](size_t completed, size_t total) {
        if (report) report({"weights", completed, total});
    };
    if (mode == LoadMode::mapped) {
        std::vector<size_t> warmed;
        for (size_t i : every) if (!external[i]) warmed.push_back(i);
        detail::warm(file, warmed, progress);
    }
    WeightPlan plan;
    plan.resident = resident;
    plan.composite = composite;
    plan.made_host.assign(file.tensors.size(), 0);
    const size_t devices = backends.size();
    loaded->backends = backends;
    const auto built = std::chrono::steady_clock::now();
    // The mapped mode copies each weight as the model resolves its role; the streamed ones give every copied weight storage first and stream the copies after.
    const AdoptWeight adopt = planning_adopt(weights, devices, plan, mode != LoadMode::mapped, make, release);
    // The fit, then the model built: a model with lookup tables was fitted above, beside its tables, and any other is fitted here.
    // Each build starts on an empty upload plan: a build given up for fewer KV tokens drops its records before the devices' memory is read again (build_fitted).
    const auto allocating = [&] {
        plan.uploads.clear();
        plan.host_reads.assign(weights.tensors.size(), 0);
        if (report) report({"allocate"});
    };
    if (!placement && report) report({"fit"});
    PlacedModel placed = placement ? build_fitted(weights, std::move(*placement), request, adopt, allocating)
                                   : place_model(weights, std::move(backends), request, options, adopt, allocating);
    // The host's composites are made side by side on as many threads as the host computes on, largest first, before any weight is streamed.
    // Their readers open first, since reader_of opens them lazily and the fill reads from several threads.
    if (!unfilled.empty()) {
        int threads = 1;
        for (const auto& b : loaded->backends)
            if (b->reads_in_place()) threads = std::max(threads, b->threads_available());
        for (size_t i : unfilled)
            for (const auto& part : file.composites.at(i).parts) {
                const size_t offset = part.second, bytes = size_t(part.first.data_size());
                if (!file.part_data(offset, bytes)) reader_of(file_index(file.part_span(offset, bytes).file));
            }
        std::stable_sort(unfilled.begin(), unfilled.end(), [&](size_t x, size_t y) { return file.tensor_bytes(x) > file.tensor_bytes(y); });
        core::JobThreads(std::min(size_t(threads), unfilled.size()) - 1).run(unfilled.size(), [&](size_t k) { fill(unfilled[k]); });
        unfilled.clear();
    }
    loaded->times.construct = detail::seconds_since(built);
    loaded->plan = std::move(placed.plan);
    loaded->dtype = std::move(placed.dtype);
    loaded->checkpoint_kv_tokens = placed.checkpoint_kv_tokens;
    loaded->kv_kept_context = placed.kv_kept_context;
    loaded->model = std::move(placed.model);
    if (const auto join = loaded->model->parts_join()) loaded->chat.flat = chat::FlatParts{join->first, loaded->tok->vocab.at(join->second)};
    // The model filled its tables from the sources as it was built and keeps no view of them.
    for (size_t i : source_tensors) weights.tensors[i].data = nullptr;
    sources.clear();
    if (mode != LoadMode::mapped) {
        // Every weight has storage now, so a model that cannot be placed has failed before any weight is uploaded.
        // The payload the backends take, each tensor read once: copied weights streamed in file order, host-read ones read into the host's copy (direct), from which a device that also takes one is written, or warmed from the mapping after (auto).
        std::vector<char> copied(file.tensors.size(), 0);
        for (const Upload& u : plan.uploads) copied[u.tensor] = 1;
        auto in_file_order = [&](std::vector<size_t>& v) {
            std::sort(v.begin(), v.end(), [&](size_t a, size_t b) { return file.offsets[a] != file.offsets[b] ? file.offsets[a] < file.offsets[b] : a < b; });
        };
        std::vector<size_t> streamed, hosted, warmed, made;
        for (size_t i : every) {
            if (resident[i]) continue;
            if (composite[i]) {
                if (copied[i]) made.push_back(i);
                continue;
            }
            if (plan.host_reads[i] && mode == LoadMode::direct) hosted.push_back(i);
            else if (copied[i]) streamed.push_back(i);
            else if (plan.host_reads[i]) warmed.push_back(i);
        }
        in_file_order(streamed);
        in_file_order(hosted);
        if (mode == LoadMode::direct) {
            const size_t bytes = gguf::bytes_of(file, hosted);
            const auto available = core::host_memory_available();
            if (available && bytes > *available)
                throw std::runtime_error("--load-mode direct: the host reads " + detail::gib(bytes) + " of weights in place and has " + detail::gib(*available) +
                                         " available; auto and mapped map them instead");
        }
        const size_t total = gguf::bytes_of(file, streamed) + gguf::bytes_of(file, hosted) + gguf::bytes_of(file, warmed) + gguf::bytes_of(file, made);
        size_t done = 0;
        if (progress && total) progress(0, total);
        // auto reads around the file cache only when the load would not fit in it and the file system takes direct reads.
        const bool around = mode == LoadMode::automatic && detail::reads_around(gguf::bytes_of(file, streamed), core::host_memory_available());
        std::vector<size_t> file_of(file.tensors.size(), 0);
        std::vector<format::FileSpan> spans(file.tensors.size());
        std::vector<char> streamed_file(paths.size(), 0);
        for (const auto* list : {&streamed, &hosted})
            for (size_t i : *list) {
                spans[i] = file.span(i);
                const size_t f = file_of[i] = file_index(spans[i].file);
                if (streamed_file[f]) continue;
                streamed_file[f] = 1;
                if (!readers[f] || (around && !readers[f]->direct())) readers[f] = detail::open_reader(paths[f], around);
                gguf::check_size(file, paths[f], readers[f]->size());
            }
        // A composite's parts are read by the stream's readers, so their files' readers open here.
        for (size_t i : made)
            for (const auto& part : file.composites.at(i).parts) {
                const size_t f = file_index(file.part_span(part.second, size_t(part.first.data_size())).file);
                if (streamed_file[f]) continue;
                streamed_file[f] = 1;
                if (!readers[f] || (around && !readers[f]->direct())) readers[f] = detail::open_reader(paths[f], around);
                gguf::check_size(file, paths[f], readers[f]->size());
            }
        std::vector<size_t> granules(paths.size(), core::page_size());
        for (size_t f = 0; f < paths.size(); ++f)
            if (readers[f]) {
                granules[f] = readers[f]->granule();
                if (streamed_file[f]) ++(readers[f]->direct() ? loaded->times.direct_files : loaded->times.files);
            }
        std::vector<std::vector<const Upload*>> destinations(file.tensors.size());
        for (const Upload& u : plan.uploads) destinations[u.tensor].push_back(&u);
        // 16 MiB pieces (docs/STATUS-2026-09.md, Loader in one place).
        auto pieces = detail::plan_pieces(streamed, spans, file_of, granules, size_t(16) << 20);
        auto into_host = detail::plan_pieces(hosted, spans, file_of, granules, size_t(16) << 20);
        for (detail::Piece& p : into_host) {
            try {
                loaded->host[p.file]->commit(p.offset, p.bytes);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(std::string("--load-mode direct: ") + e.what() + " for the weights the host reads in place; auto and mapped map them instead");
            }
            p.into = loaded->host[p.file]->data() + p.offset;
        }
        // One pass front to back over each file, the host's reads among the devices', and each composite a piece of its own at its first part, its slot sized by the bytes made.
        pieces.insert(pieces.end(), into_host.begin(), into_host.end());
        for (size_t i : made) {
            const auto& first = file.composites.at(i).parts.front();
            const format::FileSpan at = file.part_span(first.second, size_t(first.first.data_size()));
            const size_t bytes = file.tensor_bytes(i);
            detail::Piece p{file_index(at.file), at.offset, bytes, {{i, 0, 0, bytes}}};
            p.make = [&file, &readers, &file_index, i](uint8_t* into) {
                detail::make_composite(file, i, [&](const std::string& path) -> const format::FileReader& { return *readers[file_index(path)]; }, into, false);
            };
            pieces.push_back(std::move(p));
        }
        std::stable_sort(pieces.begin(), pieces.end(), [](const detail::Piece& a, const detail::Piece& b) {
            return a.file != b.file ? a.file < b.file : a.offset < b.offset;
        });
        // Each file's bytes for the report, the parts the stream sends from it.
        std::vector<size_t> file_total(paths.size(), 0), file_done(paths.size(), 0);
        for (const detail::Piece& p : pieces)
            for (const detail::Piece::Part& part : p.parts) file_total[p.file] += part.bytes;
        detail::stream(pieces, readers, destinations, [&](size_t f, size_t bytes) {
            done += bytes;
            file_done[f] += bytes;
            if (report) report({"weights", done, total, paths[f], file_done[f], file_total[f]});
        }, loaded->times);
        readers.clear();
        // A direct read spans whole granules and the gaps between near tensors, so the pages of a host's copy that hold none of its weights go back, keeping a page after each weight as a mapping's neighbours are there to read.
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> keep(paths.size());
        for (size_t i : hosted) keep[file_of[i]].push_back({spans[i].offset, spans[i].offset + spans[i].bytes + core::page_size()});
        for (const detail::Piece& p : into_host) {
            uint64_t at = p.offset;
            const uint64_t end = p.offset + p.bytes;
            for (const auto& [lo, hi] : keep[p.file]) {
                if (hi <= at) continue;
                if (lo >= end) break;
                if (lo > at) loaded->host[p.file]->decommit(size_t(at), size_t(lo - at));
                at = std::max(at, hi);
            }
            if (end > at) loaded->host[p.file]->decommit(size_t(at), size_t(end - at));
        }
        if (!warmed.empty()) {
            const size_t before = done;
            detail::warm(file, warmed, [&](size_t completed, size_t) {
                if (completed && progress) progress(before + completed, total);
            });
        } else if (progress && !total) {
            progress(0, 0);
        }
    }
    // With no host reading a weight the host's copy goes, and otherwise the tensors no host reads leave its working set.
    if (std::find(plan.host_reads.begin(), plan.host_reads.end(), 1) == plan.host_reads.end()) {
        file.release_payload();
        loaded->host.clear();
    } else {
        for (size_t i = 0; i < plan.host_reads.size(); ++i)
            if (!plan.host_reads[i]) file.drop_pages(i);
    }
    return loaded;
}

} // namespace detail

// Read a GGUF file or a Hugging Face directory (inference/directory.hpp) and its optional paired drafter, then load its weights through the shared placement and I/O path.
// `history`, where given, is the tokens of the one history the caller holds, read through the tokenizer before the fit, which budgets it (PlacementRequest::histories).
inline std::unique_ptr<LoadedModel> load_model(const std::string& path, std::vector<backend::BackendPtr> backends,
                                               const PlacementRequest& request, const ModelOptions& options = {},
                                               const LoadReport& report = {}, LoadMode mode = LoadMode{},
                                               const std::string& drafter_file = {}, const std::function<size_t(const LoadedModel&)>& history = {}) {
    auto loaded = std::make_unique<LoadedModel>();
    if (report) report({"metadata"});
    const bool directory = is_model_directory(path);
    loaded->file = read_model(path);
    if (!options.chat_template.empty()) loaded->replaced_template = chat::replace_template(loaded->file, options.chat_template);
    PlacementRequest placing = request;
    if (!drafter_file.empty()) {
        gguf::GGUFModel side = gguf::read_gguf(drafter_file);
        loaded->drafter = spec::pair(loaded->file, path, side, drafter_file);
        if (*loaded->drafter == spec::DrafterKind::mtp) {
            spec::join_blocks(loaded->file, std::move(side));
            placing.drafter = true;
        }
    }
    loaded->tok.emplace(loaded->file);
    loaded->chat = chat::chat_format(loaded->file, *loaded->tok);
    if (history) {
        placing.histories = 1;
        placing.history_tokens = history(*loaded);
    }
    ModelWeights weights = gguf_weights(loaded->file, request.rope);
    return detail::load_weights(std::move(loaded), std::move(weights), std::move(backends), placing, options, report, mode, directory);
}

} // namespace infer
