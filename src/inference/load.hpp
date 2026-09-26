#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/host_memory.hpp"
#include "format/file_reader.hpp"
#include "format/gguf.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/chat.hpp"
#include "model/arch_qwen.hpp"

// Loading a model file, the one sequence every command and tool opens a model through (docs/src/inference-load.md).

namespace infer {

// How a load reads the weights, the CLI's --load-mode, the same on every backend; the first is the default.
// `automatic` streams the weights a copying backend takes in large reads on two reader threads, overlapped with the uploads, around the file cache when they are more than the host can cache, and maps the files only for the weights a host reads in place.
// `mapped` maps the files, reads every page in before the model is placed, and copies the weights out of the mapping as the model is built.
// `direct` streams every weight around the file cache, a host's own into memory laid out as the file, and maps nothing.
enum class LoadMode { automatic, mapped, direct };

inline LoadMode load_mode_of(const std::string& name) {
    if (name == "auto") return LoadMode::automatic;
    if (name == "mapped") return LoadMode::mapped;
    if (name == "direct") return LoadMode::direct;
    throw std::runtime_error("unknown load mode '" + name + "' (auto, mapped or direct)");
}
inline const char* load_mode_name(LoadMode m) { return m == LoadMode::mapped ? "mapped" : m == LoadMode::direct ? "direct" : "auto"; }

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
    std::string plan;                // what each device of a split was given (LayerSplit::describe), empty otherwise
    LoadTimes times;
    std::vector<core::HostPages> host;   // a direct load's copy of each file, laid out as the file, for the weights a host reads in place
    std::unique_ptr<Model> model;

    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
};

// A weight a copying backend took: the storage the model's construction allocated for it (Backend::alloc_weight), filled from the file once the model is built.
struct Upload {
    size_t tensor = 0;
    backend::Backend* backend = nullptr;
    backend::BufferPtr buffer;
};

// What the loader's hook records while the model is built: per tensor whether a backend that reads in place took it, and every weight a copying backend took.
struct WeightPlan {
    std::vector<char> host_reads;
    std::vector<Upload> uploads;
};

// The adoption hook load_model builds the model with: a backend that reads in place adopts a weight where it lies and plan.host_reads[i] is set.
// A backend that copies adopts the weight there and then unless `defer`; with `defer` it gets storage the loader streams the weight into once the model is built (Backend::alloc_weight), recorded in plan.uploads.
// The model hands each tensor to each of `backends` at most once, and the records are sized for that before the model is built, so recording cannot fail while a buffer is held.
inline AdoptWeight planning_adopt(const QwenWeights& weights, size_t backends, WeightPlan& plan, bool defer) {
    plan.host_reads.assign(weights.tensors.size(), 0);
    plan.uploads.clear();
    plan.uploads.reserve(defer ? weights.tensors.size() * backends : 0);
    return [&weights, &plan, defer](size_t i, backend::Backend& b) {
        const TensorView& t = weights.tensors[i];
        if (b.reads_in_place()) {
            plan.host_reads[i] = 1;
            return b.adopt(t.data, t.bytes);
        }
        if (!defer) return b.adopt(t.data, t.bytes);
        backend::BufferPtr buffer = b.alloc_weight(t.bytes);
        plan.uploads.push_back({i, &b, buffer});
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
// Pages read in stay resident only while the host can hold them: tensors larger than the memory available would be evicted before whoever reads them does, and read from disk twice, so they are left to be read once, and the progress goes straight to complete.
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
// A piece with `into` is read straight there, into a direct load's copy of the file the host reads in place, and nothing is written from it.
struct Piece {
    struct Part {
        size_t tensor = 0, tensor_offset = 0, piece_offset = 0, bytes = 0;
    };
    size_t file = 0;
    uint64_t offset = 0;
    size_t bytes = 0;
    std::vector<Part> parts;
    uint8_t* into = nullptr;
};

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

// Read `pieces` on two reader threads into a ring of four slots, reader j taking pieces j, j + 2 and so on, or straight to a piece's `into`, and on this thread send each part to every upload of its tensor (`destinations`) in order, then report its bytes to `streamed`.
// A backend that reads a slot in place (Backend::wrap_host) takes a part as one device copy out of the slot, and the slot goes back to the readers once the copies out of it retire; any other takes it as a write, which consumes it at once, as do the parts of an `into` piece.
// A read that comes up short of a part means the file was cut after its header was read.
// Whatever fails, the readers are stopped and joined, the copies out of the ring retired and the ring freed before the error goes on; the uploads' storage stays with the model, which drains its backends before it frees any.
inline void stream(const std::vector<Piece>& pieces, const std::vector<std::unique_ptr<format::FileReader>>& readers,
                   const std::vector<std::vector<const Upload*>>& destinations, const std::function<void(size_t)>& streamed,
                   LoadTimes& times) {
    if (pieces.empty()) return;
    constexpr size_t kSlots = 4, kReaders = 2;
    size_t slot_bytes = 0;
    for (const Piece& p : pieces) slot_bytes = std::max(slot_bytes, p.bytes);
    std::vector<core::HostPages> ring;
    for (size_t i = 0; i < std::min(kSlots, pieces.size()); ++i) ring.emplace_back(slot_bytes);
    const size_t slots = ring.size(), threads = std::min(kReaders, slots);

    // The backends the parts go to, each with its view of every slot where it reads them in place, and per slot the ticket of its last copies out of it.
    // The views are declared after the ring, so they go before it.
    std::vector<backend::Backend*> targets;
    for (const auto& list : destinations)
        for (const Upload* u : list)
            if (std::find(targets.begin(), targets.end(), u->backend) == targets.end()) targets.push_back(u->backend);
    std::vector<std::vector<backend::BufferPtr>> views(targets.size());
    for (size_t t = 0; t < targets.size(); ++t)
        for (core::HostPages& slot : ring) {
            backend::BufferPtr v = targets[t]->wrap_host(slot.data(), slot.size());
            if (!v) {
                views[t].clear();
                break;
            }
            views[t].push_back(std::move(v));
        }
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

    std::mutex mu;
    std::condition_variable cv;
    size_t consumed = 0;                    // pieces whose slots are free again
    std::vector<size_t> ready(slots, 0);    // per slot, 1 + the piece read into it, 0 while none is
    std::vector<size_t> got(slots, 0);
    std::vector<double> reading(threads, 0);
    bool stop = false;
    std::exception_ptr failed;

    std::vector<std::thread> pool;
    auto halt = [&] {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : pool) t.join();
    };
    try {
        for (size_t j = 0; j < threads; ++j) {
            pool.emplace_back([&, j] {
                try {
                    for (size_t k = j; k < pieces.size(); k += threads) {
                        {
                            std::unique_lock<std::mutex> lock(mu);
                            cv.wait(lock, [&] { return stop || k < consumed + slots; });
                            if (stop) return;
                        }
                        const Piece& p = pieces[k];
                        const auto t0 = std::chrono::steady_clock::now();
                        const size_t n = readers[p.file]->read(p.offset, p.into ? p.into : ring[k % slots].data(), p.bytes);
                        const double dt = seconds_since(t0);
                        {
                            std::lock_guard<std::mutex> lock(mu);
                            got[k % slots] = n;
                            ready[k % slots] = k + 1;
                            reading[j] += dt;
                        }
                        cv.notify_all();
                    }
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        if (!failed) failed = std::current_exception();
                    }
                    cv.notify_all();
                }
            });
        }
        for (size_t k = 0; k < pieces.size(); ++k) {
            size_t n = 0;
            {
                const auto t0 = std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return failed || ready[k % slots] == k + 1; });
                times.wait += seconds_since(t0);
                if (ready[k % slots] != k + 1) std::rethrow_exception(failed);
                n = got[k % slots];
            }
            const Piece& p = pieces[k];
            const size_t slot = k % slots;
            const uint8_t* data = p.into ? p.into : ring[slot].data();
            const auto t0 = std::chrono::steady_clock::now();
            size_t bytes = 0;
            for (const Piece::Part& part : p.parts) {
                if (part.piece_offset + part.bytes > n)
                    throw std::runtime_error(readers[p.file]->path() + " ended at " + std::to_string(p.offset + n) + " bytes, before its tensors");
                for (const Upload* u : destinations[part.tensor]) {
                    const size_t t = target_of(u->backend);
                    if (!p.into && !views[t].empty()) {
                        u->backend->copy(*u->buffer, part.tensor_offset, *views[t][slot], part.piece_offset, part.bytes);
                        copying[slot][t] = 1;
                        times.copied += part.bytes;
                    } else {
                        u->backend->write(*u->buffer, part.tensor_offset, data + part.piece_offset, part.bytes);
                    }
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
            {
                std::lock_guard<std::mutex> lock(mu);
                consumed = held ? k : k + 1;
            }
            cv.notify_all();
            if (streamed) streamed(bytes);
        }
        const auto t0 = std::chrono::steady_clock::now();
        retire((pieces.size() - 1) % slots);
        times.upload += seconds_since(t0);
    } catch (...) {
        halt();
        for (backend::Backend* b : targets) b->sync();
        throw;
    }
    halt();
    times.read = *std::max_element(reading.begin(), reading.end());
}

} // namespace detail

// Load the model at `path`, a GGUF file or the first shard of a set, over the caller's `backends` as `request` places it, reading the weights as `mode` says: read the headers, build the tokenizer and the chat format, place the model, and fill the weights the backends copy, reporting the payload to `progress`.
// The host's copy of the weights is then released when no host reads one in place, and otherwise the pages of every tensor no host reads leave its working set.
inline std::unique_ptr<LoadedModel> load_model(const std::string& path, std::vector<backend::BackendPtr> backends,
                                               const PlacementRequest& request, const ModelOptions& options = {},
                                               const format::LoadProgress& progress = {}, LoadMode mode = LoadMode{}) {
    auto loaded = std::make_unique<LoadedModel>();
    gguf::GGUFModel& file = loaded->file;
    loaded->times.mode = mode;
    file = gguf::read_gguf(path);
    std::vector<size_t> every(file.tensors.size());
    for (size_t i = 0; i < every.size(); ++i) every[i] = i;
    const bool host = host_reads_in_place(backends, request);
    // The files, in the payload's order, and a reader for each once one is opened; a direct load opens them all first, so a file system that does not take direct reads is refused before anything is built.
    std::vector<std::string> paths;
    for (const auto& seg : file.segments)
        if (paths.empty() || paths.back() != seg.path) paths.push_back(seg.path);
    std::vector<std::unique_ptr<format::FileReader>> readers(paths.size());
    auto file_index = [&](const std::string& p) { return size_t(std::find(paths.begin(), paths.end(), p) - paths.begin()); };
    if (mode == LoadMode::direct) {
        for (size_t f = 0; f < paths.size(); ++f) {
            try {
                readers[f] = std::make_unique<format::FileReader>(paths[f], true);
            } catch (const format::DirectUnavailable& e) {
                throw std::runtime_error(std::string("--load-mode direct: ") + e.what());
            }
            gguf::check_size(file, paths[f], readers[f]->size());
        }
    }
    // A mapped load reads every page in before the model is placed; auto maps the files only for a host, and direct never.
    if (mode == LoadMode::mapped || (mode == LoadMode::automatic && host)) gguf::map_payload(file);
    if (mode == LoadMode::mapped) detail::warm(file, every, progress);
    loaded->tok.emplace(file);
    loaded->chat = chat::chat_format(file, *loaded->tok);
    QwenWeights weights = gguf_weights(file);
    // A direct load gives a host a copy of each file laid out as the file, so every weight keeps the offset within its page the mapping would give it: address space now, memory once the model says which weights the host reads.
    if (mode == LoadMode::direct && host) {
        // Each copy covers its file as its header gave it, rounded up to the reads' granule, since the last read ends on it.
        for (size_t f = 0; f < paths.size(); ++f) {
            const uint64_t g = readers[f]->granule();
            loaded->host.push_back(core::HostPages::reserved(size_t((gguf::file_size(file, paths[f]) + g - 1) / g * g)));
        }
        for (size_t i : every) {
            const format::FileSpan s = file.span(i);
            weights.tensors[i].data = loaded->host[file_index(s.file)].data() + s.offset;
        }
    }
    WeightPlan plan;
    const size_t devices = backends.size();
    const auto built = std::chrono::steady_clock::now();
    // The mapped mode copies each weight as the model resolves its role; the streamed ones give every copied weight storage first and stream the copies after.
    PlacedModel placed = place_model(weights, std::move(backends), request, options,
                                     planning_adopt(weights, devices, plan, mode != LoadMode::mapped));
    loaded->times.construct = detail::seconds_since(built);
    loaded->plan = std::move(placed.plan);
    loaded->model = std::move(placed.model);
    if (mode != LoadMode::mapped) {
        // Every weight has storage now, so a model that cannot be placed has failed before any weight is uploaded.
        // The payload the backends take, each tensor read once: the copied weights streamed in file order, and those a host reads read into its copy (direct), from which a device that also takes one is written, or warmed from the mapping after (auto).
        std::vector<char> copied(file.tensors.size(), 0);
        for (const Upload& u : plan.uploads) copied[u.tensor] = 1;
        auto in_file_order = [&](std::vector<size_t>& v) {
            std::sort(v.begin(), v.end(), [&](size_t a, size_t b) { return file.offsets[a] != file.offsets[b] ? file.offsets[a] < file.offsets[b] : a < b; });
        };
        std::vector<size_t> streamed, hosted, warmed;
        for (size_t i : every) {
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
        const size_t total = gguf::bytes_of(file, streamed) + gguf::bytes_of(file, hosted) + gguf::bytes_of(file, warmed);
        size_t done = 0;
        if (progress && total) progress(0, total);
        // auto reads around the file cache only when the load would not fit in it and the file system takes direct reads.
        const bool around = mode == LoadMode::automatic && detail::reads_around(gguf::bytes_of(file, streamed), core::host_memory_available());
        std::vector<size_t> file_of(file.tensors.size(), 0);
        std::vector<format::FileSpan> spans(file.tensors.size());
        for (const auto* list : {&streamed, &hosted})
            for (size_t i : *list) {
                spans[i] = file.span(i);
                const size_t f = file_of[i] = file_index(spans[i].file);
                if (readers[f]) continue;
                readers[f] = detail::open_reader(paths[f], around);
                gguf::check_size(file, paths[f], readers[f]->size());
            }
        std::vector<size_t> granules(paths.size(), core::page_size());
        for (size_t f = 0; f < paths.size(); ++f)
            if (readers[f]) {
                granules[f] = readers[f]->granule();
                ++(readers[f]->direct() ? loaded->times.direct_files : loaded->times.files);
            }
        std::vector<std::vector<const Upload*>> destinations(file.tensors.size());
        for (const Upload& u : plan.uploads) destinations[u.tensor].push_back(&u);
        // 16 MiB pieces (docs/STATUS.md, the loader's step 5).
        auto pieces = detail::plan_pieces(streamed, spans, file_of, granules, size_t(16) << 20);
        auto into_host = detail::plan_pieces(hosted, spans, file_of, granules, size_t(16) << 20);
        for (detail::Piece& p : into_host) {
            try {
                loaded->host[p.file].commit(p.offset, p.bytes);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(std::string("--load-mode direct: ") + e.what() + " for the weights the host reads in place; auto and mapped map them instead");
            }
            p.into = loaded->host[p.file].data() + p.offset;
        }
        // One pass front to back over each file, the host's reads among the devices'.
        pieces.insert(pieces.end(), into_host.begin(), into_host.end());
        std::stable_sort(pieces.begin(), pieces.end(), [](const detail::Piece& a, const detail::Piece& b) {
            return a.file != b.file ? a.file < b.file : a.offset < b.offset;
        });
        detail::stream(pieces, readers, destinations, [&](size_t bytes) {
            done += bytes;
            if (progress) progress(done, total);
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
                if (lo > at) loaded->host[p.file].decommit(size_t(at), size_t(lo - at));
                at = std::max(at, hi);
            }
            if (end > at) loaded->host[p.file].decommit(size_t(at), size_t(end - at));
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

} // namespace infer
