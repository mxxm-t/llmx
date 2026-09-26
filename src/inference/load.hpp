#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
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

// How a load reads the weights a copying backend takes, the CLI's --load-mode, the same on every backend.
// `automatic` streams them from the files in large reads on two reader threads, overlapped with the uploads, and maps the files only for the weights a host reads in place.
// `mapped` maps the files, reads every page in before the model is placed, and copies the weights out of the mapping.
enum class LoadMode { automatic, mapped };

inline LoadMode load_mode_of(const std::string& name) {
    if (name == "auto") return LoadMode::automatic;
    if (name == "mapped") return LoadMode::mapped;
    throw std::runtime_error("unknown load mode '" + name + "' (auto or mapped)");
}
inline const char* load_mode_name(LoadMode m) { return m == LoadMode::mapped ? "mapped" : "auto"; }

// Where a load's time went, in seconds, for the CLI's timing line: building the model, and for a streamed load the busiest reader thread's reads, the uploads, and the uploads' waits for a read.
struct LoadTimes {
    LoadMode mode = LoadMode::automatic;
    size_t files = 0;       // files the weights were streamed from
    size_t streamed = 0;    // bytes read for them, gaps and granule rounding included
    double construct = 0, read = 0, upload = 0, wait = 0;
};

// A model file loaded for use: the file, its tokenizer and chat format, and the model placed over the caller's backends.
// A host backend's buffers read the file's payload in place while the model lives, so load_model builds this in place and it is never copied or moved; `model` is declared last, so it is destroyed first.
struct LoadedModel {
    gguf::GGUFModel file;
    std::optional<bpe::Tokenizer> tok;
    chat::ChatFormat chat;
    std::string plan;                // what each device of a split was given (LayerSplit::describe), empty otherwise
    LoadTimes times;
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

// One read of the stream: `bytes` from `offset` in file `file`, both on the file's granule, and the parts of tensors it holds.
struct Piece {
    struct Part {
        size_t tensor = 0, tensor_offset = 0, piece_offset = 0, bytes = 0;
    };
    size_t file = 0;
    uint64_t offset = 0;
    size_t bytes = 0;
    std::vector<Part> parts;
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

// Read `pieces` on `readers_count` reader threads into a ring of four slots, reader j taking pieces j, j + readers_count and so on, and on this thread write each part to every upload of its tensor (`destinations`) in order, then report its bytes to `streamed`.
// A read that comes up short of a part means the file was cut after its header was read.
// Whatever fails, the readers are stopped and joined and the ring freed before the error goes on; the uploads' storage stays with the model, which drains its backends before it frees any.
inline void stream(const std::vector<Piece>& pieces, const std::vector<std::unique_ptr<format::FileReader>>& readers,
                   const std::vector<std::vector<const Upload*>>& destinations, const std::function<void(size_t)>& streamed,
                   LoadTimes& times, size_t readers_count = 1) {
    if (pieces.empty()) return;
    constexpr size_t kSlots = 4;
    size_t slot_bytes = 0;
    for (const Piece& p : pieces) slot_bytes = std::max(slot_bytes, p.bytes);
    std::vector<core::HostPages> ring;
    for (size_t i = 0; i < std::min(kSlots, pieces.size()); ++i) ring.emplace_back(slot_bytes);
    const size_t slots = ring.size(), threads = std::max<size_t>(1, std::min(readers_count, slots));

    std::mutex mu;
    std::condition_variable cv;
    size_t consumed = 0;                    // pieces this thread has written out
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
                        const size_t n = readers[p.file]->read(p.offset, ring[k % slots].data(), p.bytes);
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
            const uint8_t* data = ring[k % slots].data();
            const auto t0 = std::chrono::steady_clock::now();
            size_t bytes = 0;
            for (const Piece::Part& part : p.parts) {
                if (part.piece_offset + part.bytes > n)
                    throw std::runtime_error(readers[p.file]->path() + " ended at " + std::to_string(p.offset + n) + " bytes, before its tensors");
                for (const Upload* u : destinations[part.tensor])
                    u->backend->write(*u->buffer, part.tensor_offset, data + part.piece_offset, part.bytes);
                bytes += part.bytes;
            }
            times.upload += seconds_since(t0);
            times.streamed += p.bytes;
            {
                std::lock_guard<std::mutex> lock(mu);
                consumed = k + 1;
            }
            cv.notify_all();
            if (streamed) streamed(bytes);
        }
    } catch (...) {
        halt();
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
                                               const format::LoadProgress& progress = {}, LoadMode mode = LoadMode::automatic) {
    auto loaded = std::make_unique<LoadedModel>();
    gguf::GGUFModel& file = loaded->file;
    loaded->times.mode = mode;
    file = gguf::read_gguf(path);
    std::vector<size_t> every(file.tensors.size());
    for (size_t i = 0; i < every.size(); ++i) every[i] = i;
    // A mapped load reads every page in before the model is placed; a streamed one maps the files only when a host reads weights in place, which with experts on the CPU is the CPU backend the placement adds.
    const bool host = request.cpu_moe != 0 ||
                      std::any_of(backends.begin(), backends.end(), [](const backend::BackendPtr& b) { return b->reads_in_place(); });
    if (mode == LoadMode::mapped || host) gguf::map_payload(file);
    if (mode == LoadMode::mapped) detail::warm(file, every, progress);
    loaded->tok.emplace(file);
    loaded->chat = chat::chat_format(file, *loaded->tok);
    const QwenWeights weights = gguf_weights(file);
    WeightPlan plan;
    const size_t devices = backends.size();
    const auto built = std::chrono::steady_clock::now();
    // The mapped mode copies each weight as the model resolves its role, as reading from a mapping always did; the streamed one plans every weight first and streams the copies after.
    PlacedModel placed = place_model(weights, std::move(backends), request, options,
                                     planning_adopt(weights, devices, plan, mode == LoadMode::automatic));
    loaded->times.construct = detail::seconds_since(built);
    loaded->plan = std::move(placed.plan);
    loaded->model = std::move(placed.model);
    if (mode == LoadMode::automatic) {
        // Every weight has storage now, so a model that cannot be placed has failed before any weight is uploaded.
        // The payload the backends take, each tensor once: the copied weights streamed in file order, then the pages of those a host alone reads warmed.
        std::vector<char> copied(file.tensors.size(), 0);
        for (const Upload& u : plan.uploads) copied[u.tensor] = 1;
        std::vector<size_t> streamed, warmed;
        for (size_t i : every) {
            if (copied[i]) streamed.push_back(i);
            else if (plan.host_reads[i]) warmed.push_back(i);
        }
        std::sort(streamed.begin(), streamed.end(), [&](size_t a, size_t b) { return file.offsets[a] != file.offsets[b] ? file.offsets[a] < file.offsets[b] : a < b; });
        const size_t total = gguf::bytes_of(file, streamed) + gguf::bytes_of(file, warmed);
        size_t done = 0;
        if (progress && total) progress(0, total);
        // One reader per file, and each tensor's destinations.
        std::vector<std::unique_ptr<format::FileReader>> readers;
        std::vector<size_t> file_of(file.tensors.size(), 0), granules;
        std::vector<format::FileSpan> spans(file.tensors.size());
        for (size_t i : streamed) {
            spans[i] = file.span(i);
            size_t f = 0;
            while (f < readers.size() && readers[f]->path() != spans[i].file) ++f;
            if (f == readers.size()) {
                readers.push_back(std::make_unique<format::FileReader>(spans[i].file));
                granules.push_back(readers.back()->granule());
            }
            file_of[i] = f;
        }
        std::vector<std::vector<const Upload*>> destinations(file.tensors.size());
        for (const Upload& u : plan.uploads) destinations[u.tensor].push_back(&u);
        // Two readers and 16 MiB pieces measured fastest of one or two readers and 16 or 32 MiB (docs/STATUS.md, the loader's step 5).
        const auto pieces = detail::plan_pieces(streamed, spans, file_of, granules, size_t(16) << 20);
        loaded->times.files = readers.size();
        detail::stream(pieces, readers, destinations, [&](size_t bytes) {
            done += bytes;
            if (progress) progress(done, total);
        }, loaded->times, 2);
        readers.clear();
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
    } else {
        for (size_t i = 0; i < plan.host_reads.size(); ++i)
            if (!plan.host_reads[i]) file.drop_pages(i);
    }
    return loaded;
}

} // namespace infer
