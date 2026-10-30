#pragma once
// The routes of docs/SERVER.md: native /v1/generate, /v1/chat, /v1/tokenize, /v1/detokenize and /v1/health, and the OpenAI-compatible /v1/chat/completions, /v1/completions and /v1/models.
// Both families share one parse, one scheduler request and one drain loop; one thread per connection parses, tokenizes, submits and drains.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include "core/json.hpp"
#include "config.hpp"
#include "core/utf8.hpp"
#include "inference/chat.hpp"
#include "inference/load.hpp"
#include "inference/media.hpp"
#include "model/place.hpp"
#include "quant/types.hpp"
#include "server/context_cut.hpp"
#include "server/http.hpp"
#include "server/scheduler.hpp"

namespace server {

// The host memory for donors the devices evict when the server is given none (--host-cache-bytes): what `max_seqs` conversations take at the most one request may hold (`limit`, Scheduler::request_limit), within half of what is free.
// Each copy still leaves the host the pool's fixed reserve (Model::host_admits).
inline size_t default_host_cache(const infer::Model& model, size_t max_seqs, size_t limit) {
    if (!model.caches_on_devices()) return 0;
    const size_t bt = model.kv_block_tokens();
    // The donors' share of half the host memory free once the model is loaded: what the lookup tables' hot rows leave of it (infer::host_shares), none where they take it all.
    const std::optional<size_t> free = core::host_memory_available();
    const size_t half = free ? *free / 2 : 0, rows = model.lookup_store_bytes();
    const std::optional<size_t> cap = free ? std::optional<size_t>(half > rows ? half - rows : 0) : std::nullopt;
    return host_cache_default(model.host_bytes(limit / bt * bt), max_seqs, cap);
}

// The start line for uncapped requests that fill the pool and will not take turns here, or not all of them, though turns were possible or asked for; empty where they will, and on the CPU with no host memory given.
// The host memory is asked first: given enough, a server takes turns wherever its caches are (docs/SERVER.md, turns at the pool's edge).
inline std::string no_turns(const infer::Model& model, size_t host_cap, bool given, size_t limit) {
    const size_t bt = model.kv_block_tokens(), need = model.host_bytes(limit / bt * bt);
    if (!given && !model.caches_on_devices()) return "";
    const size_t tokens = host_cap ? turn_tokens(host_cap, bt, limit, [&](size_t n) { return model.host_bytes(n); }) : 0;
    return turns_line(host_cap, need, tokens, model.keeps_state(), model.checkpoint_slots());
}

struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    size_t max_seqs = 16;
    size_t max_queue = 64;   // requests waiting for admission; past it, 503
    size_t passes = 0;       // passes in flight; 0 takes the stages and one more on a pipelined layer split, S + 1, and one elsewhere (Scheduler)
    int state_checkpoints = -1;   // on a model that keeps a state, the states kept for prefix reuse; -1 for the most the fit gives up to max_seqs
    bool timing = false;     // time the rounds and the stages for /v1/health, over backends made to time their work
    std::optional<size_t> host_cache_bytes;   // host memory for donors the devices evict (Scheduler); none given takes default_host_cache
    DiskOptions disk;                         // the disk tier under the host tier (docs/DISK-TIER.md); its bytes 0 keeps none
    std::string model_name;
    infer::DtypePlan dtype;
    // The proposer a decoding request drafts with and its most drafts a verify (docs/SPECULATIVE.md, section 3); none drafts nothing.
    infer::spec::Proposer* proposer = nullptr;
    std::vector<std::string> drafter_sources;   // the sources `--drafter` listed in the order tried, which /v1/health names
    size_t draft_max = 0;
    bool context_shift = false;   // a prompt that does not fit is cut at a turn's end (--context-overflow shift) and not refused
    size_t ctx_size = 0;          // the tokens a request may hold that serve's --ctx-size asks, 0 for the model's context (Scheduler::request_limit)
    // The vision encoder `--mmproj` opened, and its file's digest, which an image's digest begins with (docs/VISION.md, sections 5 and 6); none takes no images.
    infer::Vision* vision = nullptr;
    std::array<uint8_t, 32> vision_identity{};
};

// The most bytes an image of a request decodes to from its data URL (docs/VISION.md, section 6).
inline constexpr size_t kMaxImageBytes = size_t(20) << 20;

// Strict base64: the standard alphabet, padded to whole quadruples, nothing else; empty on any other text.
inline std::optional<std::string> base64_decode(const std::string& text) {
    if (text.size() % 4) return std::nullopt;
    std::string out;
    out.reserve(text.size() / 4 * 3);
    const auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    for (size_t i = 0; i < text.size(); i += 4) {
        const bool last = i + 4 == text.size();
        const size_t pads = last ? (text[i + 3] == '=') + (text[i + 2] == '=' && text[i + 3] == '=') : 0;
        int v[4];
        for (size_t k = 0; k < 4; ++k) {
            v[k] = k >= 4 - pads ? 0 : value(text[i + k]);
            if (v[k] < 0) return std::nullopt;
        }
        const uint32_t n = uint32_t(v[0]) << 18 | uint32_t(v[1]) << 12 | uint32_t(v[2]) << 6 | uint32_t(v[3]);
        out.push_back(char(n >> 16));
        if (pads < 2) out.push_back(char(n >> 8 & 0xff));
        if (pads < 1) out.push_back(char(n & 0xff));
    }
    return out;
}

// The longest prefix of `bytes` that ends on a complete UTF-8 character, so a token whose text ends mid-character is held until the rest comes.
inline size_t utf8_complete(const std::string& bytes) {
    size_t i = bytes.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char)bytes[i - 1];
        if ((c & 0xC0) != 0x80)
            return back + 1 >= utf8::lead_length(c) ? bytes.size() : i - 1;
        --i;
        ++back;
    }
    return bytes.size();
}

// The text with each byte that starts no valid UTF-8 character replaced by U+FFFD, since a byte-level vocabulary can sample bytes that form no character and JSON carries only characters.
inline std::string utf8_sanitize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const size_t len = utf8::valid_length(s, i);
        if (len) { out.append(s, i, len); i += len; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

// An error reply's body in the native shape, or in the compatible routes' shape when `compat` is set.
// The message is repaired as generated text is, since it can carry bytes from outside the request: a chat template's text from the model file, or a backend's failure.
inline std::string error_json(const std::string& message, bool compat) {
    const std::string text = jmini::quote(utf8_sanitize(message));
    if (!compat) return "{\"error\":" + text + "}";
    return "{\"error\":{\"message\":" + text + ",\"type\":\"invalid_request_error\"}}";
}

// The least log-probability a compatible shape writes, since its fields are numbers: a lower one, minus infinity or a NaN is written as this, a probability of zero to any float.
inline constexpr float kLogprobFloor = -9999.0f;

// A log-probability in a compatible shape: the shortest decimal that reads back as the float, or kLogprobFloor for one below it or not a number.
// The native shapes write jmini::number's own, which is null where JSON has no number.
inline std::string compat_number(float logprob) {
    return jmini::number(logprob >= kLogprobFloor ? logprob : kLogprobFloor);
}

// The assistant turn a client sends back for a chat reply `text`, which the reply is read again as (Api::next_turn).
// On the compatible route it is the content alone and the calls in the format `tools` names (chat::reply_message), and on the native route the text whole, as the format keeps a turn.
inline chat::Message reply_sent_back(const chat::ChatFormat& format, bool compatible, bool opened, const std::string& text, chat::ToolFormat tools,
                                     const chat::ToolTypes& typed, bool single, bool* open) {
    if (!compatible) return format.assistant(text);
    return chat::reply_message(format, opened, tools, typed, single, text, open);
}

// Cuts a reply's text at the first stop text, which the compatible routes leave out, and holds back the end of the text so far that could still grow into one.
class StopTrim {
public:
    explicit StopTrim(const std::vector<std::string>& stops) : stops_(stops) {}
    // The part of `piece`, with what was held, that no stop text can still reach.
    std::string feed(const std::string& piece) {
        if (hit_) return {};
        held_ += piece;
        size_t at = std::string::npos;
        for (const std::string& s : stops_)
            if (!s.empty()) at = std::min(at, held_.find(s));
        if (at != std::string::npos) {
            hit_ = true;
            std::string out = held_.substr(0, at);
            held_.clear();
            return out;
        }
        size_t keep = 0;
        for (const std::string& s : stops_)
            for (size_t len = std::min(s.size() - 1, held_.size()); len > keep; --len)
                if (held_.compare(held_.size() - len, len, s, 0, len) == 0) { keep = len; break; }
        std::string out = held_.substr(0, held_.size() - keep);
        held_.erase(0, held_.size() - keep);
        return out;
    }
    // What is still held when the reply ended without a stop text.
    std::string finish() {
        std::string out;
        out.swap(held_);
        return out;
    }

private:
    const std::vector<std::string>& stops_;
    std::string held_;
    bool hit_ = false;
};

// The record of a server's start, which /v1/health gives while it starts and keeps once it serves (docs/SERVER.md, Startup): each phase reached, its state, seconds and bytes.
class Startup {
public:
    explicit Startup(std::string model) : model_(utf8_sanitize(model)), started_((int64_t)std::time(nullptr)), t0_(std::chrono::steady_clock::now()) {}

    // A step of the start: a phase the CLI, the loader (infer::LoadStep) or the server begins, which ends the one running, or the bytes a phase has read.
    void step(const infer::LoadStep& s) {
        std::lock_guard<std::mutex> lk(m_);
        const auto now = std::chrono::steady_clock::now();
        if (phases_.empty() || phases_.back().name != s.phase) {
            if (!phases_.empty()) close(phases_.back(), "done", now);
            phases_.push_back(Phase{s.phase, "running", now, 0, 0, 0, {}});
        }
        Phase& p = phases_.back();
        if (!s.total) return;
        p.completed = s.completed, p.total = s.total;
        if (s.file.empty()) return;
        const std::string name = utf8_sanitize(std::filesystem::u8path(s.file).filename().u8string());
        auto f = std::find_if(p.files.begin(), p.files.end(), [&](const File& x) { return x.name == name; });
        if (f == p.files.end()) f = p.files.insert(p.files.end(), File{name});
        f->completed = s.file_completed, f->total = s.file_total;
    }
    void begin(const char* phase) { step(infer::LoadStep{phase}); }
    // The start has ended: the server serves its routes.
    void ready() {
        std::lock_guard<std::mutex> lk(m_);
        if (!phases_.empty()) close(phases_.back(), "done", std::chrono::steady_clock::now());
        ready_ = true;
        ended_ = std::chrono::steady_clock::now();
    }
    // The start failed with `error`, which the record keeps until the process ends.
    void fail(const std::string& error) {
        std::lock_guard<std::mutex> lk(m_);
        if (!phases_.empty()) close(phases_.back(), "failed", std::chrono::steady_clock::now());
        error_ = utf8_sanitize(error);
        ended_ = std::chrono::steady_clock::now();
    }
    // The `startup` object of /v1/health.
    std::string json() const {
        std::lock_guard<std::mutex> lk(m_);
        const auto now = std::chrono::steady_clock::now();
        const auto n = [](uint64_t v) { return std::to_string(v); };
        std::string phases;
        for (const Phase& p : phases_) {
            const double s = p.state == "running" ? seconds(p.begun, now) : p.seconds;
            phases += std::string(phases.empty() ? "" : ",") + "{\"name\":" + jmini::quote(p.name) + ",\"state\":\"" + p.state + "\",\"seconds\":" + jmini::number(float(s));
            if (p.total) {
                phases += ",\"bytes_done\":" + n(p.completed) + ",\"bytes_total\":" + n(p.total) + ",\"bytes_per_s\":" + n(s > 0 ? uint64_t(double(p.completed) / s) : 0) + ",\"files\":[";
                for (size_t i = 0; i < p.files.size(); ++i)
                    phases += std::string(i ? "," : "") + "{\"name\":" + jmini::quote(p.files[i].name) + ",\"bytes_done\":" + n(p.files[i].completed) + ",\"bytes_total\":" + n(p.files[i].total) + "}";
                phases += "]";
            }
            phases += "}";
        }
        const std::string phase = !error_.empty() ? "failed" : ready_ ? "ready" : phases_.empty() ? "" : phases_.back().name;
        return "{\"started_unix\":" + std::to_string(started_) + ",\"elapsed_s\":" + jmini::number(float(seconds(t0_, ended_ ? *ended_ : now))) + ",\"phase\":" + jmini::quote(phase) +
               (error_.empty() ? std::string() : ",\"error\":" + jmini::quote(error_)) + ",\"phases\":[" + phases + "]}";
    }
    // /v1/health's body while the server starts: its status, the server's and the requests' objects in their shapes, and the record.
    std::string starting() const {
        bool failed;
        {
            std::lock_guard<std::mutex> lk(m_);
            failed = !error_.empty();
        }
        const std::string numerics = std::string(LLMX_NUMERICS).substr(0, 16);
        return std::string("{\"status\":\"") + (failed ? "failed" : "starting") + "\",\"server\":{\"version\":" + jmini::quote(LLMX_VERSION_STRING) + ",\"numerics\":" + jmini::quote(numerics) +
               ",\"uptime_s\":" + std::to_string(std::max<int64_t>(0, (int64_t)std::time(nullptr) - started_)) + ",\"model\":" + jmini::quote(model_) +
               "},\"requests\":{\"now\":{\"active\":0,\"queued\":0,\"paused\":0,\"kv_tokens\":0}},\"startup\":" + json() + "}";
    }

private:
    struct File {
        std::string name;
        size_t completed = 0, total = 0;
    };
    struct Phase {
        std::string name, state;
        std::chrono::steady_clock::time_point begun;
        double seconds = 0;
        size_t completed = 0, total = 0;
        std::vector<File> files;
    };
    static double seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
    static void close(Phase& p, const char* state, std::chrono::steady_clock::time_point now) {
        if (p.state != "running") return;
        p.state = state;
        p.seconds = seconds(p.begun, now);
    }
    const std::string model_;
    const int64_t started_;
    const std::chrono::steady_clock::time_point t0_;
    mutable std::mutex m_;
    std::vector<Phase> phases_;   // under m_
    std::string error_;           // under m_
    bool ready_ = false;          // under m_
    std::optional<std::chrono::steady_clock::time_point> ended_;   // under m_, when the start was ready or failed
};

class Front;

class Api {
public:
    Api(infer::Model& model, const bpe::Tokenizer& tok, const chat::ChatFormat& format, Scheduler& sched,
        const Config& cfg, const Startup* startup = nullptr)
        : model_(model), tok_(tok), format_(format), sched_(sched), cfg_(cfg),
          cached_topology_json_(topology_json(model_.topology(), cfg_.dtype)), startup_(startup), started_((int64_t)std::time(nullptr)) {
        // The model's name is its file's, which on Linux can hold bytes that are not UTF-8, and every reply that names the model carries it.
        cfg_.model_name = utf8_sanitize(cfg_.model_name);
    }

    void handle(http::Connection& c) {
        http::Request req;
        int status = 0;
        http::Limits limits;
        const bool read = c.read_request(req, limits, status);
        // The compatible routes report errors in their clients' shape, a request refused while it is read included; the path is known unless the head itself was refused.
        const std::optional<Route> route = route_of(req.path);
        const bool shape = route && compat(*route);
        if (!read) {
            if (status) c.respond(status, "application/json", error_json(http::reason(status), shape));
            return;
        }
        try {
            if (req.method == "GET" && req.path == "/v1/health") return health(c);
            if (req.method == "GET" && req.path == "/v1/live") return c.respond(200, "application/json", "{\"status\":\"ok\"}");
            if (req.method == "GET" && req.path == "/v1/models") return models(c);
            if (req.method == "POST" && req.path == "/v1/tokenize") return tokenize(c, req);
            if (req.method == "POST" && req.path == "/v1/detokenize") return detokenize(c, req);
            if (req.method == "POST" && route) return generate(c, req, *route);
            c.respond(404, "application/json", error_json("no such route", shape));
        } catch (const BadRequest& e) {
            c.respond(e.status, "application/json", error_json(e.what(), shape));
        } catch (const http::ClientGone&) {
            // A client that has left gets no answer, the 500 below included: a client that shut only its sending side may still be reading, and an error would tell it of a failure that is not the server's.
        } catch (const std::exception& e) {
            try { c.respond(500, "application/json", error_json(e.what(), shape)); } catch (...) {}
        }
    }

private:
    friend class Front;
    // How often a connection waiting on its request looks for a departed client.
    static constexpr std::chrono::milliseconds kProbe{100};
    // A chat reply's next turn is given to the scheduler again every this many tokens while it is written (Scheduler::follow).
    static constexpr size_t kFollowEvery = 32;
    // The most tokens a request may list beside each sampled one, `top_logprobs` or the completions route's `logprobs`: the compatible chat API's limit, which the completions route takes too, though that API stops at 5.
    static constexpr int kTopLogprobs = 20;

    enum class Route { generate, chat, chat_completions, completions };
    static std::optional<Route> route_of(const std::string& path) {
        if (path == "/v1/generate") return Route::generate;
        if (path == "/v1/chat") return Route::chat;
        if (path == "/v1/chat/completions") return Route::chat_completions;
        if (path == "/v1/completions") return Route::completions;
        return std::nullopt;
    }
    // The routes that speak the OpenAI clients' shape.
    static bool compat(Route route) { return route == Route::chat_completions || route == Route::completions; }

    struct BadRequest : std::runtime_error {
        int status;
        BadRequest(int s, const std::string& m) : std::runtime_error(m), status(s) {}
    };

    // The statistics, grouped as docs/SERVER.md has them: what is true now beside what has been counted since the server started.
    void health(http::Connection& c) {
        const Scheduler::Stats s = sched_.stats();
        const auto n = [](uint64_t v) { return std::to_string(v); };
        // The cache's moves by kind (Scheduler::moved).
        const auto moves = [&](const Scheduler::Stats& st) {
            std::string out;
            for (size_t k = 0; k < st.moves.size(); ++k) out += std::string(k ? "," : "") + "\"" + move_name((Move)k) + "\":" + n(st.moves[k]);
            return out;
        };
        const auto yes = [](bool v) { return std::string(v ? "true" : "false"); };
        // The disk tier's digest of the model's files and its adoption of a kept directory, which run once the server listens (DiskTier::Start).
        const auto digest_json = [&](const DiskTier::Start& d) {
            const double s = d.seconds;
            return "{\"state\":\"" + d.digest + "\",\"file\":" + jmini::quote(utf8_sanitize(d.file)) + ",\"bytes_done\":" + n(d.done) + ",\"bytes_total\":" + n(d.total) +
                   ",\"bytes_per_s\":" + n(s > 0 ? uint64_t(double(d.done) / s) : 0) + ",\"seconds\":" + jmini::number(float(s)) + "},\"adopt\":{\"state\":\"" + d.adopt + "\",\"entries\":" + n(d.adopted) + "}";
        };
        // What restores cost, each read and promotion once (Scheduler::Stats::RestoreTotals): counts, and each part's sum and largest in whole milliseconds.
        const auto restore_json = [&](const Scheduler::Stats::RestoreTotals& x) {
            const auto ms = [&](double v) { return n((uint64_t)v); };
            return ",\"restore\":{\"since_start\":{\"reads\":" + n(x.reads) + ",\"promotions\":" + n(x.promotions) + ",\"copies\":" + n(x.copies) + ",\"from_host\":" + n(x.from_host) +
                   ",\"from_disk\":" + n(x.from_disk) + ",\"timeouts\":" + n(x.timeouts) + ",\"read_ms\":" + ms(x.read_ms) + ",\"read_largest_ms\":" + ms(x.read_largest_ms) +
                   ",\"slabs_ms\":" + ms(x.slabs_ms) + ",\"slabs_largest_ms\":" + ms(x.slabs_largest_ms) + ",\"hold_ms\":" + ms(x.hold_ms) + ",\"hold_largest_ms\":" + ms(x.hold_largest_ms) +
                   ",\"enqueue_ms\":" + ms(x.enqueue_ms) + ",\"enqueue_largest_ms\":" + ms(x.enqueue_largest_ms) + ",\"copies_ms\":" + ms(x.copies_ms) + ",\"copies_largest_ms\":" + ms(x.copies_largest_ms) + "}}";
        };
        std::string devices, positions, sources, live;
        for (size_t i = 0; i < s.live.count; ++i) {
            const auto& r = s.live.items[i];
            live += (i ? "," : "") + std::string("{\"id\":") + jmini::quote("r-" + n(r.id)) +
                    ",\"phase\":" + jmini::quote(r.phase) + ",\"prompt_tokens\":" + n(r.prompt_tokens) +
                    ",\"reused_tokens\":" + n(r.reused_tokens) + ",\"prefilled_tokens\":" + n(r.prefilled_tokens) +
                    ",\"published_tokens\":" + n(r.published_tokens) + ",\"elapsed_ms\":" + n(r.elapsed_ms) + ",\"steps\":" + n(r.steps) + ",\"verifies\":" + n(r.verifies) +
                    ",\"drafted\":" + n(r.drafted) + ",\"kept\":" + n(r.kept) +
                    ",\"queued_ms\":" + n(r.queued_ms) + ",\"first_token_ms\":" + n(r.first_token_ms) + ",\"inter_token_ms\":" + jmini::number((float)r.inter_token_ms) +
                    (r.image_index ? ",\"image\":{\"index\":" + n(r.image_index) + ",\"count\":" + n(r.image_count) + ",\"rows\":" + n(r.image_rows) +
                                         ",\"pieces_done\":" + n(r.pieces_done) + ",\"pieces_total\":" + n(r.pieces_total) + ",\"elapsed_ms\":" + n(r.image_elapsed_ms) + "}"
                                   : std::string()) + "}";
        }
        size_t drafted = 0, kept = 0, own_drafted = 0, own_kept = 0;
        for (size_t x : s.own_drafted) own_drafted += x;
        for (size_t x : s.own_kept) own_kept += x;
        for (const auto& source : cfg_.drafter_sources) sources += (sources.empty() ? "" : ",") + jmini::quote(source);
        for (const auto& dev : cfg_.dtype.devices) devices += (devices.empty() ? "" : ",") + jmini::quote(dev.name);
        for (size_t i = 0; i < s.drafted.size(); ++i) {
            const size_t k = i < s.kept.size() ? s.kept[i] : 0;
            positions += (i ? "," : "") + std::string("{\"position\":") + n(i + 1) + ",\"drafted\":" + n(s.drafted[i]) + ",\"kept\":" + n(k) + "}";
            drafted += s.drafted[i];
            kept += k;
        }
        const std::string numerics = std::string(LLMX_NUMERICS).substr(0, 16);
        const int64_t up = (int64_t)std::time(nullptr) - started_;
        c.respond(200, "application/json",
                  "{\"status\":\"ok\",\"server\":{\"version\":" + jmini::quote(LLMX_VERSION_STRING) + ",\"numerics\":" + jmini::quote(numerics) +
                  ",\"uptime_s\":" + n((uint64_t)std::max<int64_t>(0, up)) + ",\"model\":" + jmini::quote(cfg_.model_name) +
                  ",\"context_tokens\":" + n(model_.context_length()) + ",\"request_tokens\":" + n(sched_.token_limit()) + ",\"kv_tokens\":" + n(model_.kv_tokens_total()) + ",\"devices\":[" + devices + "]}" + (startup_ ? ",\"startup\":" + startup_->json() : std::string()) +
                  ",\"topology\":" + cached_topology_json_ + ",\"precision\":" + dtype_json(cfg_.dtype) +
                  ",\"requests\":{\"now\":{\"active\":" + n(s.active) + ",\"queued\":" + n(s.queued) + ",\"paused\":" + n(s.paused) + ",\"kv_tokens\":" + n(s.kv_tokens) +
                  "},\"limits\":{\"active\":" + n(cfg_.max_seqs) + ",\"queued\":" + n(cfg_.max_queue) +
                  "},\"since_start\":{\"finished\":" + n(s.finished) + ",\"prompt_tokens\":" + n(s.prompt_tokens) + ",\"generated_tokens\":" + n(s.generated_tokens) +
                  ",\"shifts\":" + n(shifts_.load()) + ",\"shifted_tokens\":" + n(shifted_tokens_.load()) +
                  "},\"live\":{\"sample_elapsed_ms\":" + n(s.live.sample_elapsed_ms) + ",\"limit\":" + n(Scheduler::LiveRequests::limit) +
                  ",\"total\":" + n(s.live.total) + ",\"omitted\":" + n(s.live.total - s.live.count) +
                  ",\"work\":{\"prefilled_tokens\":" + n(s.live.prefilled_tokens) + ",\"generated_tokens\":" + n(s.live.published_tokens) + "},\"items\":[" + live + "]}}" +
                  ",\"work\":{\"since_start\":{\"prefilled_tokens\":" + n(s.prefilled_tokens) + ",\"generated_tokens\":" + n(s.published_tokens) + "}}" +
                  ",\"reuse\":{\"since_start\":{\"forks\":" + n(s.prefix_hits) + ",\"tokens\":" + n(s.prefix_tokens) +
                  "},\"device\":{\"now\":{\"entries\":" + n(s.donors) + ",\"state_checkpoints\":" + n(s.checkpoints) + ",\"kv_tokens\":" + n(s.kv_donor_tokens) + "}}" +
                  ",\"host\":{\"now\":{\"entries\":" + n(s.host_donors) + ",\"bytes\":" + n(s.host_bytes) + ",\"limit_bytes\":" + n(s.host_limit) + ",\"allocated_bytes\":" + n(s.host_allocated) +
                  "},\"since_start\":{\"promotions\":" + n(s.host_hits) + ",\"yielded_bytes\":" + n(s.host_yielded) + ",\"bytes_moved\":" + n(s.host_bytes_moved) + "}}" +
                  ",\"disk\":{\"now\":{\"entries\":" + n(s.disk_entries) + ",\"segments\":" + n(s.disk_entries - s.disk_states) + ",\"states\":" + n(s.disk_states) + ",\"bytes\":" + n(s.disk_bytes) + ",\"limit_bytes\":" + n(s.disk_limit) +
                  ",\"unproved\":" + n(s.disk_unproved) + ",\"in_flight\":" + n(s.disk_in_flight) + ",\"ready\":" + yes(s.disk_ready) + ",\"writing\":" + yes(s.disk_writing) + ",\"digest\":" + digest_json(s.disk_digest) +
                  "},\"since_start\":{\"hits\":" + n(s.disk_hits) + ",\"bytes_read\":" + n(s.disk_bytes_read) + ",\"bytes_written\":" + n(s.disk_bytes_written) +
                  ",\"waits\":" + n(s.disk_waits) + ",\"wait_timeouts\":" + n(s.disk_wait_timeouts) + ",\"wait_ms\":" + n((uint64_t)s.disk_wait_ms) + ",\"errors\":" + n(s.disk_errors) +
                  ",\"refused\":" + n(s.disk_refused) + ",\"dropped_for_cap\":" + n(s.disk_capped) + ",\"proved\":" + n(s.disk_proved) + ",\"disproved\":" + n(s.disk_disproved) + ",\"lost_before_written\":" + n(s.lost_before_written) + ",\"refused_directories\":" + n(s.disk_refused_dirs) +
                  ",\"room_waits\":" + n(s.room_waits) + ",\"room_wait_timeouts\":" + n(s.room_wait_timeouts) + ",\"room_wait_ms\":" + n((uint64_t)s.room_wait_ms) +
                  ",\"room_wait_longest_ms\":" + n((uint64_t)s.room_wait_longest_ms) + "}}" +
                  ",\"boundaries\":{\"now\":{\"entries\":" + n(s.boundaries) + "},\"since_start\":{\"hits\":" + n(s.boundary_hits) + "}}" + misses_json(s) + restore_json(s.restore) + ",\"moves\":{\"since_start\":{" + moves(s) + "}}}" +
                  ",\"pressure\":{\"since_start\":{\"pauses\":" + n(s.pauses) + ",\"stalls\":" + n(s.stalls) + ",\"waits\":" + n(s.waits) +
                  ",\"recomputed_tokens\":" + n(s.recomputed) + ",\"resumes_taking_history_back\":" + n(s.taken_back) + ",\"turns\":" + n(s.yields) + ",\"turns_without_memory\":" + n(s.turns_without_memory) + "}}" +
                  ",\"reread\":{\"now\":{\"jobs\":" + n(s.jobs) + ",\"rows_left\":" + n(s.job_rows_left) + "},\"since_start\":{\"jobs\":" + n(s.reprefills) + ",\"rows\":" + n(s.reprefill_rows) + ",\"cancelled\":" + n(s.reprefill_cancels) + ",\"handed\":" + n(s.reprefill_handed) + ",\"rows_released\":" + n(s.reprefill_released) + "}}" +
                  ",\"drafting\":{\"sources\":[" + sources + "],\"since_start\":{\"drafted\":" + n(drafted) + ",\"kept\":" + n(kept) + ",\"failed\":" + n(s.draft_failures) + ",\"by_position\":[" + positions + "]" +
                  ",\"own_history\":{\"drafted\":" + n(own_drafted) + ",\"kept\":" + n(own_kept) + "}}}" +
                  ",\"passes\":{\"limit\":" + n(s.passes) + ",\"in_flight\":" + n(s.in_flight) + ",\"sampling_threads\":" + n(s.samplers) + ",\"since_start\":{\"prompt\":" + n(s.prompt_passes) + ",\"continued\":" + n(s.continued) + ",\"rule_replies\":" + n(s.rule_replies) + ",\"rule_chunks\":" + n(s.rule_chunks) + ",\"rule_mixed\":" + n(s.rule_mixed) + ",\"holds\":" + n(s.holds) + ",\"stage_waits\":" + n(s.stage_waits) + "}}" +
                  ",\"service\":" + service_json(s.service) +
                  (s.timed ? ",\"timing\":" + timing_json(s.timing) : std::string()) + "}");
    }
    // The service record (docs/SERVER.md, The price table): per stage the passes measured against their predictions, the last few, and the prices by band and rows; the rows by kind; and the requests' waits, first tokens and gaps between tokens.
    static std::string service_json(const ServiceRecord::Snapshot& v) {
        const auto n = [](uint64_t x) { return std::to_string(x); };
        const auto ms = [](double x) { return jmini::number((float)x); };
        const auto list = [](const std::vector<std::string>& items) {
            std::string out = "[";
            for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + items[i];
            return out + "]";
        };
        const auto work = [&](const PassWork& w) {
            std::vector<std::string> chunks;
            for (const PassWork::Chunk& c : w.chunks) chunks.push_back("{\"class\":" + n(c.cls) + ",\"band\":" + n(c.band) + ",\"rows\":" + n(c.rows) + "}");
            return ",\"decode_rows\":" + n(w.decode_rows) + ",\"decode_band\":" + n(w.decode_band) + ",\"chunks\":" + list(chunks);
        };
        // A run of cells of one key, their rows, mean, last measure and count as arrays.
        struct Run {
            std::string head;
            std::vector<std::string> rows, mean, last, samples;
            std::string json() const {
                const auto join = [](const std::vector<std::string>& xs) {
                    std::string out = "[";
                    for (size_t i = 0; i < xs.size(); ++i) out += (i ? "," : "") + xs[i];
                    return out + "]";
                };
                return "{" + head + ",\"rows\":" + join(rows) + ",\"ms\":" + join(mean) + ",\"last_ms\":" + join(last) + ",\"samples\":" + join(samples) + "}";
            }
        };
        std::vector<std::string> stages;
        for (const StageRecord& st : v.stages) {
            std::vector<std::string> recent, decode, chunks;
            for (const StageRecord::Pass& p : st.recent)
                recent.push_back("{\"pass\":" + n(p.pass) + ",\"measured_ms\":" + ms(p.measured_ms) + ",\"predicted_ms\":" + ms(p.predicted_ms) + work(p.work) + "}");
            Run run;
            const auto flush = [&](std::vector<std::string>& to) {
                if (!run.rows.empty()) to.push_back(run.json());
                run = Run{};
            };
            size_t band = SIZE_MAX;
            for (const auto& [key, cell] : st.prices.decode) {
                if (key.first != band) {
                    flush(decode);
                    band = key.first;
                    run.head = "\"band\":" + n(band);
                }
                run.rows.push_back(n(key.second));
                run.mean.push_back(ms(cell.ms));
                run.last.push_back(ms(cell.last_ms));
                run.samples.push_back(n(cell.samples));
            }
            flush(decode);
            std::pair<size_t, size_t> at{SIZE_MAX, SIZE_MAX};
            for (const auto& [key, cell] : st.prices.chunks) {
                const size_t cls = std::get<0>(key), b = std::get<1>(key);
                if (std::make_pair(cls, b) != at) {
                    flush(chunks);
                    at = {cls, b};
                    run.head = "\"class\":" + n(cls) + ",\"extent\":" + n(cls < v.extents.size() ? v.extents[cls] : 0) + ",\"band\":" + n(b);
                }
                run.rows.push_back(n(std::get<2>(key)));
                run.mean.push_back(ms(cell.ms));
                run.last.push_back(ms(cell.last_ms));
                run.samples.push_back(n(cell.samples));
            }
            flush(chunks);
            stages.push_back("{\"since_start\":{\"passes\":" + n(st.passes) + ",\"predicted\":" + n(st.predicted) + ",\"over_prediction\":" + n(st.over) +
                             ",\"measured_ms\":" + n((uint64_t)st.measured_ms) + ",\"predicted_ms\":" + n((uint64_t)st.predicted_ms) + ",\"error_ms\":" + n((uint64_t)st.error_ms) +
                             ",\"longest_ms\":" + ms(st.longest_ms) + ",\"recording_ms\":" + n((uint64_t)st.recording_ms) + ",\"lookup_wait_ms\":" + n((uint64_t)st.lookup_ms) +
                             ",\"sum_wait_ms\":" + n((uint64_t)st.sum_ms) + ",\"after_recording_ms\":" + n((uint64_t)st.after_ms) + "},\"recent\":" + list(recent) + ",\"prices\":{\"decode\":" + list(decode) + ",\"chunks\":" + list(chunks) + "}}");
        }
        std::vector<std::string> bands;
        for (uint64_t x : v.rows.by_band) bands.push_back(n(x));
        const auto summary = [&](const Recent::Summary& q) {
            return "{\"count\":" + n(q.count) + ",\"p50\":" + ms(q.p50) + ",\"p95\":" + ms(q.p95) + ",\"p99\":" + ms(q.p99) + ",\"max\":" + ms(q.max) + "}";
        };
        return "{\"stages\":" + list(stages) + ",\"rows\":{\"since_start\":{\"decode\":" + n(v.rows.decode) + ",\"drafts\":" + n(v.rows.drafts) + ",\"recomputed\":" + n(v.rows.recomputed) +
               ",\"prompt\":" + n(v.rows.prompt) + ",\"reread\":" + n(v.rows.reread) + ",\"prompt_by_band\":" + list(bands) + "}},\"requests\":{\"queue_wait_ms\":" + summary(v.queue_wait) +
               ",\"first_token_ms\":" + summary(v.first_token) + ",\"inter_token_ms\":" + summary(v.inter_token) + "},\"objective\":{\"reply_tokens_per_s\":" + jmini::number(kReplyTokensPerSecond) +
               ",\"max_step_gap_ms\":" + jmini::number(kMaxStepGapMs) + ",\"gap_ms\":" + ms(v.objective.gap_ms) + ",\"since_start\":{\"met\":" + n(v.objective.met) + ",\"unmet\":" + n(v.objective.unmet) +
               ",\"capped\":" + n(v.objective.capped) + "}}}";
    }
    static std::string topology_json(const infer::Model::Topology& t, const infer::DtypePlan& dtype) {
        const auto n = [](size_t v) { return std::to_string(v); };
        const auto members = [&](const std::vector<size_t>& ids) {
            std::string out = "[";
            for (size_t i = 0; i < ids.size(); ++i) out += (i ? "," : "") + n(ids[i]);
            return out + "]";
        };
        const char* mode = t.mode == infer::Model::Topology::Mode::single ? "single" :
                           t.mode == infer::Model::Topology::Mode::layer ? "layer" :
                           t.mode == infer::Model::Topology::Mode::tensor ? "tensor" : "staged_tensor";
        std::string devices, stages, layers, lookups;
        for (const auto& d : t.devices) {
            const std::string name = d.index < dtype.devices.size() ? dtype.devices[d.index].name : "device " + n(d.index);
            devices += (devices.empty() ? "" : ",") + std::string("{\"index\":") + n(d.index) + ",\"name\":" + jmini::quote(name) +
                       ",\"used\":" + (d.used ? "true" : "false") + ",\"member\":" + n(d.member) + "}";
        }
        for (const auto& s : t.stages)
            stages += (stages.empty() ? "" : ",") + std::string("{\"index\":") + n(s.index) + ",\"first_layer\":" + n((size_t)s.first_layer) +
                      ",\"end_layer\":" + n((size_t)s.end_layer) + ",\"members\":" + members(s.members) + "}";
        for (const auto& l : t.layers) {
            const std::string stream = l.stream_members.empty() ? "null" :
                std::string("{\"members\":") + members(l.stream_members) + ",\"min_prompt_tokens\":" + n(l.stream_min_prompt_tokens) + "}";
            layers += (layers.empty() ? "" : ",") + std::string("{\"index\":") + n(l.index) + ",\"mixer_members\":" + members(l.mixer_members) +
                      ",\"ffn_members\":" + members(l.ffn_members) + ",\"routed\":" + (l.routed ? "true" : "false") +
                      ",\"expert_stream\":" + stream + "}";
        }
        for (const auto& lookup : t.lookups) {
            const quant::StorageType* type = quant::storage_type(lookup.type);
            lookups += (lookups.empty() ? "" : ",") + std::string("{\"layer\":") + n(lookup.layer) + ",\"rows\":" + n(lookup.rows) +
                       ",\"row_bytes\":" + n(lookup.row_bytes) + ",\"bytes\":" + n(lookup.bytes) +
                       ",\"type\":" + jmini::quote(type ? type->name : std::to_string(lookup.type)) +
                       ",\"storage\":" + jmini::quote(lookup.file_backed ? "file" : "device") + ",\"members\":" + members(lookup.members) + "}";
        }
        return std::string("{\"mode\":\"") + mode + "\",\"mixed_roles\":" + (t.mixed_roles ? "true" : "false") +
               ",\"tensor_width\":" + n(t.tensor_width) + ",\"devices\":[" + devices + "],\"stages\":[" + stages +
               "],\"embedding_members\":" + members(t.embedding_members) + ",\"head_members\":" + members(t.head_members) + ",\"layers\":[" + layers +
               "],\"lookups\":[" + lookups + "]}";
    }
    // The run's request and resolved dtype, including each device's emulation or wider fallback.
    static std::string dtype_json(const infer::DtypePlan& d) {
        std::string devices;
        for (const auto& dev : d.devices)
            devices += (devices.empty() ? "" : ",") + std::string("{\"device\":") + jmini::quote(dev.name) + ",\"how\":" + jmini::quote(dev.how) + ",\"paths\":" + jmini::quote(dev.paths) + ",\"effective\":\"" + backend::dtype_name(dev.effective) + "\"}";
        return std::string("{\"requested\":\"") + d.requested_name() + "\",\"declared\":\"" + backend::dtype_name(d.declared) + "\",\"effective\":\"" +
               backend::dtype_name(d.effective) + "\",\"devices\":[" + devices + "]}";
    }
    // A timed scheduler's figures (--timing): each thread time as a mean over the rounds, each stage's idle share over the span its device time was read in, and the device-bound rate of that span.
    static std::string timing_json(const Scheduler::Timing& t) {
        const double n = t.rounds ? (double)t.rounds : 1.0;
        const auto mean = [n](double ms) { return jmini::number((float)(ms / n)); };
        std::string idle;
        double busiest = 0;
        for (size_t s = 0; s < t.stage_ms.size() && t.span_ms > 0; ++s) {
            idle += (idle.empty() ? "" : ",") + jmini::number((float)std::max(0.0, 1.0 - t.stage_ms[s] / t.span_ms));
            busiest = std::max(busiest, t.stage_ms[s]);
        }
        return "{\"rounds\":" + std::to_string(t.rounds) + ",\"round_ms\":" + mean(t.round_ms) + ",\"recording_ms\":" + mean(t.recording_ms) +
               ",\"relaying_ms\":" + mean(t.relaying_ms) + ",\"sampling_ms\":" + mean(t.sampling_ms) + ",\"assembly_ms\":" + mean(t.assembly_ms) +
               ",\"drafting_ms\":" + mean(t.drafting_ms) + ",\"settling_ms\":" + mean(t.settling_ms) +
               ",\"receive_wait_ms\":" + mean(t.receive_wait_ms) + ",\"staging_wait_ms\":" + mean(t.staging_wait_ms) +
               ",\"open_wait_ms\":" + mean(t.open_wait_ms) + ",\"logits_wait_ms\":" + mean(t.logits_wait_ms) + ",\"stage_idle_share\":[" + idle +
               "],\"device_bound_rows_per_s\":" + jmini::number((float)(busiest > 0 ? 1000.0 * (double)t.rows / busiest : 0.0)) + "}";
    }
    // The list clients read the model id from, with the file's context length and vocabulary beside the standard fields.
    void models(http::Connection& c) {
        c.respond(200, "application/json",
                  "{\"object\":\"list\",\"data\":[{\"id\":" + jmini::quote(cfg_.model_name) +
                  ",\"object\":\"model\",\"created\":" + std::to_string(started_) + ",\"owned_by\":\"llmx\"" +
                  ",\"context_length\":" + std::to_string(model_.context_length()) +
                  ",\"vocab\":" + std::to_string(model_.n_vocab()) + "}]}");
    }

    // A sampling setting within its range in infer::Sampling (inference/sampler.hpp), the range the CLI reads its flag against, so both refuse the same values.
    // A number past the float's range is refused before the cast, which has no defined result for it, and the range is checked on the float the sampler reads.
    static float setting(const jmini::Value& v, const char* key, const infer::SampleRange<float>& range, float fallback) {
        const jmini::Value* f = v.get(key);
        if (!f) return fallback;
        const double x = f->isNumber() ? f->asNumber() : NAN;
        if (!(std::fabs(x) <= std::numeric_limits<float>::max()) || !range.holds((float)x)) {
            char text[96];
            if (range.hi == std::numeric_limits<float>::max()) std::snprintf(text, sizeof text, " must be a number of at least %g", range.lo);
            else std::snprintf(text, sizeof text, " must be a number from %g to %g", range.lo, range.hi);
            throw BadRequest(400, key + std::string(text));
        }
        return (float)x;
    }
    // A whole number from lo to hi, refused before the caller casts it, since a double outside the target type has no defined conversion.
    static double integer_value(const jmini::Value& f, const std::string& name, double lo, double hi) {
        const double x = f.isNumber() ? f.asNumber() : NAN;
        if (!(x >= lo && x <= hi) || x != std::floor(x)) {
            char range[96];
            std::snprintf(range, sizeof range, " must be an integer from %.0f to %.0f", lo, hi);
            throw BadRequest(400, name + range);
        }
        return x;
    }
    // A whole-number field, or `fallback` when it is absent.
    static double integer(const jmini::Value& v, const char* key, double lo, double hi, double fallback) {
        const jmini::Value* f = v.get(key);
        return f ? integer_value(*f, key, lo, hi) : fallback;
    }
    static bool flag(const jmini::Value& v, const char* key) {
        const jmini::Value* f = v.get(key);
        return f && f->t == jmini::Value::T::Bool && f->b;
    }
    // A POST route's body, which must be one JSON object.
    static jmini::Value body_of(const http::Request& req) {
        jmini::Value body;
        try { body = jmini::parse(req.body); }
        catch (const std::exception& e) { throw BadRequest(400, std::string("bad JSON: ") + e.what()); }
        if (!body.isObject()) throw BadRequest(400, "the body must be a JSON object");
        return body;
    }
    static std::string ids_json(const std::vector<uint32_t>& ids) {
        std::string json = "[";
        for (size_t i = 0; i < ids.size(); ++i) json += (i ? "," : "") + std::to_string(ids[i]);
        return json + "]";
    }
    // Whether a field is present with a value other than null, which the compatible APIs send for one not set.
    static bool given(const jmini::Value& v, const char* key) {
        const jmini::Value* f = v.get(key);
        return f && f->t != jmini::Value::T::Null;
    }
    // A setting that is true or false, anything else refused rather than read as either.
    static bool boolean(const jmini::Value& v, const char* key, bool fallback) {
        const jmini::Value* f = v.get(key);
        if (!f) return fallback;
        if (f->t != jmini::Value::T::Bool) throw BadRequest(400, key + std::string(" must be true or false"));
        return f->b;
    }

    // A message's content: a string, or the array of text and image parts the compatible chat route accepts, an image as {"type": "image_url", "image_url": {"url"}} (docs/VISION.md, section 6).
    // An assistant message that carries tool_calls may leave its content null or out, which renders as an empty content; with an image, `parts` holds the content as parts.
    static std::string content_of(const jmini::Value& m, std::vector<chat::ContentPart>* parts = nullptr) {
        const jmini::Value* content = m.get("content");
        if ((!content || content->t == jmini::Value::T::Null) && given(m, "tool_calls")) return {};
        if (!content) throw BadRequest(400, "every message needs a content");
        if (content->isString()) return content->asString();
        if (!content->isArray()) throw BadRequest(400, "message content must be a string or an array of content parts");
        std::string text;
        std::vector<chat::ContentPart> all;
        bool image = false;
        for (const auto& part : content->asArray()) {
            const jmini::Value* type = part.get("type");
            if (type && type->isString() && type->asString() == "image_url") {
                (void)image_url_of(part);
                all.push_back(chat::ContentPart{true, ""});
                image = true;
                continue;
            }
            const jmini::Value* t = part.get("text");
            if (!type || !type->isString() || type->asString() != "text" || !t || !t->isString())
                throw BadRequest(400, "a content part must be a text or an image_url part");
            text += t->asString();
            all.push_back(chat::ContentPart{false, t->asString()});
        }
        if (image && parts) *parts = std::move(all);
        return text;
    }

    // An image part's data URL, its `detail` absent or "auto", the one sizing being --image-max-pixels.
    static std::string image_url_of(const jmini::Value& part) {
        const jmini::Value* image = part.get("image_url");
        const jmini::Value* url = image && image->isObject() ? image->get("url") : nullptr;
        if (!url || !url->isString()) throw BadRequest(400, "an image_url part needs image_url.url, a string");
        const jmini::Value* detail = image->get("detail");
        if (detail && detail->t != jmini::Value::T::Null && (!detail->isString() || detail->asString() != "auto"))
            throw BadRequest(400, "image_url.detail is not taken: images are sized by the server's --image-max-pixels");
        return url->asString();
    }

    // An image's file bytes from its data URL: data: alone, PNG or JPEG by its first bytes whatever the type says, strict base64, at most kMaxImageBytes decoded.
    static std::string image_bytes(const std::string& url) {
        const size_t comma = url.find(',');
        if (url.rfind("data:", 0) != 0 || comma == std::string::npos)
            throw BadRequest(400, "an image must be given as a data: URL; http, https and file URLs are not fetched");
        const std::string head = url.substr(5, comma - 5);
        if (head != "image/png;base64" && head != "image/jpeg;base64" && head != "image/jpg;base64")
            throw BadRequest(400, "an image's data URL must be image/png or image/jpeg in base64");
        if ((url.size() - comma - 1) / 4 * 3 > kMaxImageBytes) throw BadRequest(400, "an image is larger than its limit of 20 MiB");
        const std::optional<std::string> bytes = base64_decode(url.substr(comma + 1));
        if (!bytes) throw BadRequest(400, "an image's data URL is not strict base64");
        if (bytes->size() > kMaxImageBytes) throw BadRequest(400, "an image is larger than its limit of 20 MiB");
        const bool png = bytes->size() >= 8 && bytes->compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0;
        const bool jpeg = bytes->size() >= 3 && (unsigned char)(*bytes)[0] == 0xFF && (unsigned char)(*bytes)[1] == 0xD8 && (unsigned char)(*bytes)[2] == 0xFF;
        if (!png && !jpeg) throw BadRequest(400, "an image is neither PNG nor JPEG by its first bytes");
        return *bytes;
    }

    // The data URLs of a request's images in the order its prompt renders them, message by message.
    static std::vector<std::string> image_urls(const jmini::Value& body) {
        std::vector<std::string> urls;
        const jmini::Value* msgs = body.get("messages");
        if (!msgs || !msgs->isArray()) return urls;
        for (const auto& m : msgs->asArray()) {
            const jmini::Value* content = m.get("content");
            if (!content || !content->isArray()) continue;
            for (const auto& part : content->asArray()) {
                const jmini::Value* type = part.get("type");
                if (type && type->isString() && type->asString() == "image_url") urls.push_back(image_url_of(part));
            }
        }
        return urls;
    }

    // A request's images, each URL decoded, resized and hashed once on the connection's thread, the times of each logged; the scheduler encodes them, since it alone runs the backends (docs/VISION.md, section 10).
    struct RequestImages {
        std::vector<std::string> urls;
        std::map<std::string, infer::PreparedImage> prepared;
    };
    RequestImages images_of(const jmini::Value& body, Route route) const {
        RequestImages out;
        out.urls = image_urls(body);
        if (out.urls.empty()) return out;
        if (route != Route::chat && route != Route::chat_completions) throw BadRequest(400, "images are read in messages, on the chat routes");
        if (!cfg_.vision) throw BadRequest(400, "this server reads no images; start it with --mmproj");
        for (const std::string& url : out.urls) {
            if (out.prepared.count(url)) continue;
            const std::string bytes = image_bytes(url);
            try {
                const infer::PreparedImage e = infer::prepare_image(*cfg_.vision, bytes, &cfg_.vision_identity);
                std::fprintf(stderr, "server: an image of grid %ux%u, %zu rows: decoded in %.1f ms, resized in %.1f ms, hashed in %.1f ms\n", e.grid_h, e.grid_w, e.tokens,
                             e.decode_ms, e.resize_ms, e.digest_ms);
                out.prepared.emplace(url, e);
            } catch (const std::exception& e) {
                throw BadRequest(400, std::string("an image was refused: ") + e.what());
            }
        }
        return out;
    }
    // The rows of the images `body` holds now, in prompt order, a cut having dropped some.
    static std::vector<size_t> image_rows_of(const jmini::Value& body, const RequestImages& images) {
        std::vector<size_t> rows;
        for (const std::string& url : image_urls(body)) rows.push_back(images.prepared.at(url).tokens);
        return rows;
    }
    // A prompt's ids with its images' pad runs, `runs` taking each image's run where given; a body holding no image gives its ids as encode does.
    std::vector<uint32_t> prompt_ids(const jmini::Value& body, const std::string& prompt, const RequestImages& images, std::vector<infer::MediaRun>* runs = nullptr) const {
        std::vector<uint32_t> ids = encode(prompt);
        const std::vector<std::string> urls = image_urls(body);
        if (urls.empty()) return ids;
        std::vector<infer::PreparedImage> list;
        for (const std::string& url : urls) list.push_back(images.prepared.at(url));
        try {
            infer::ImagePrompt p = infer::expand(*cfg_.vision, ids, image_rows_of(body, images), &list);
            if (runs) *runs = std::move(p.runs);
            return std::move(p.ids);
        } catch (const std::exception& e) {
            throw BadRequest(400, e.what());
        }
    }
    bool is_pad(uint32_t id) const { return cfg_.vision && id == cfg_.vision->pad; }
    // `ids` with each of its pad ids made the next image's run of `rows`; none where it marks more images than `rows` gives, as a next turn's ids may not.
    std::vector<uint32_t> with_runs(const std::vector<uint32_t>& ids, const std::vector<size_t>& rows) const {
        std::vector<uint32_t> out;
        size_t next = 0;
        for (uint32_t id : ids) {
            if (!is_pad(id)) out.push_back(id);
            else if (next == rows.size()) return {};
            else out.insert(out.end(), rows[next++], id);
        }
        return out;
    }
    // `ids` with each image's pad run made one pad, as the template renders it.
    std::vector<uint32_t> one_pad(const std::vector<uint32_t>& ids) const {
        std::vector<uint32_t> out;
        for (size_t i = 0; i < ids.size(); ++i)
            if (!is_pad(ids[i]) || !i || ids[i - 1] != ids[i]) out.push_back(ids[i]);
        return out;
    }
    // How many of `ids`, which hold their images' pad runs, the first `n` ids of one_pad(ids) cover, an image's run whole.
    size_t through_runs(const std::vector<uint32_t>& ids, size_t n) const {
        size_t i = 0;
        for (size_t k = 0; k < n && i < ids.size(); ++k)
            for (const uint32_t id = ids[i++]; is_pad(id) && i < ids.size() && ids[i] == id;) ++i;
        return i;
    }
    // Where each id of `ids` ends in its text, an image's pad run taking the bytes of its one pad, so a render with one pad an image places them.
    std::vector<size_t> text_ends(const std::vector<uint32_t>& ids) const {
        std::vector<size_t> ends;
        size_t at = 0;
        for (size_t i = 0; i < ids.size(); ++i) ends.push_back(at += is_pad(ids[i]) && i && ids[i - 1] == ids[i] ? 0 : tok_.decode({ids[i]}).size());
        return ends;
    }

    // An assistant message's tool calls in the compatible chat API's form, {"id", "type": "function", "function": {"name", "arguments"}}, the arguments a JSON text or an object; none where the field is absent, null or empty.
    static std::vector<chat::ToolCall> tool_calls_of(const jmini::Value& m) {
        std::vector<chat::ToolCall> calls;
        const jmini::Value* list = m.get("tool_calls");
        if (!list || list->t == jmini::Value::T::Null) return calls;
        static const char* const form = "tool_calls must be an array of {\"type\": \"function\", \"function\": {\"name\": a string, \"arguments\": a string or an object}}";
        if (!list->isArray()) throw BadRequest(400, form);
        for (const auto& c : list->asArray()) {
            const jmini::Value* type = c.get("type");
            const jmini::Value* id = c.get("id");
            const jmini::Value* function = c.get("function");
            const jmini::Value* name = function ? function->get("name") : nullptr;
            const jmini::Value* arguments = function ? function->get("arguments") : nullptr;
            if (!c.isObject() || (type && (!type->isString() || type->asString() != "function")) || (id && !id->isString()) || !name || !name->isString() ||
                name->asString().empty() || !arguments || (!arguments->isString() && !arguments->isObject()))
                throw BadRequest(400, form);
            calls.push_back({id ? id->asString() : std::string(), name->asString(), *arguments});
        }
        return calls;
    }
    // A message's optional string field, refused when it is neither a string nor null.
    static std::optional<std::string> text_field(const jmini::Value& m, const char* key) {
        const jmini::Value* f = m.get(key);
        if (!f || f->t == jmini::Value::T::Null) return std::nullopt;
        if (!f->isString()) throw BadRequest(400, key + std::string(" must be a string"));
        return f->asString();
    }

    // A request's messages as chat records its own turns: a message's reasoning as the client passes it in `reasoning_content`, or else an assistant message as chat::ChatFormat::assistant keeps chat's own replies.
    // An assistant message's tool calls come with it, and a message's `tool_call_id` and `name` as given.
    std::vector<chat::Message> messages_of(const jmini::Value& body) const {
        const jmini::Value* msgs = body.get("messages");
        if (!msgs || !msgs->isArray() || msgs->asArray().empty()) throw BadRequest(400, "messages must be a non-empty array");
        std::vector<chat::Message> messages;
        for (const auto& m : msgs->asArray()) {
            const jmini::Value* given_role = m.get("role");
            if (!given_role || !given_role->isString()) throw BadRequest(400, "every message needs a string role");
            // The developer role of the newer chat APIs is the system message.
            const std::string role = given_role->asString() == "developer" ? "system" : given_role->asString();
            std::vector<chat::ToolCall> calls = tool_calls_of(m);
            if (!calls.empty() && role != "assistant") throw BadRequest(400, "only an assistant message may carry tool_calls");
            const jmini::Value* reasoning = m.get("reasoning_content");
            if (reasoning && reasoning->t != jmini::Value::T::Null && !reasoning->isString()) throw BadRequest(400, "reasoning_content must be a string");
            std::vector<chat::ContentPart> parts;
            if (reasoning && reasoning->isString()) messages.push_back({role, content_of(m, &parts), reasoning->asString()});
            else if (role == "assistant") messages.push_back(format_.assistant(content_of(m, &parts)));
            else messages.push_back({role, content_of(m, &parts), std::nullopt});
            messages.back().parts = std::move(parts);
            messages.back().tool_calls = std::move(calls);
            messages.back().tool_call_id = text_field(m, "tool_call_id");
            messages.back().name = text_field(m, "name");
        }
        return messages;
    }

    // A chat request's tools (docs/SERVER.md, Tool calls): given to the template where `tool_choice` is "auto", its default, and then the reply's calls parsed on the compatible route; "none" is the request without tools.
    // Refused: a strict schema, "required" or a named function, which need constrained decoding, and tools for a template without a tool-call format; `parallel_tool_calls` false is `single`, one call a reply.
    struct Tools {
        bool on = false, single = false;
        chat::jj::Value value;
        chat::ToolTypes typed;
    };
    Tools tools_of(const jmini::Value& body) const {
        Tools t;
        if (const jmini::Value* parallel = given(body, "parallel_tool_calls") ? body.get("parallel_tool_calls") : nullptr) {
            if (parallel->t != jmini::Value::T::Bool) throw BadRequest(400, "parallel_tool_calls must be true or false");
            t.single = !parallel->b;
        }
        bool none = false;
        if (const jmini::Value* choice = given(body, "tool_choice") ? body.get("tool_choice") : nullptr) {
            const std::string c = choice->isString() ? choice->asString() : std::string();
            if (c == "required") throw BadRequest(400, "tool_choice \"required\" is not supported: it needs constrained decoding; send \"auto\" or \"none\"");
            if (choice->isObject()) throw BadRequest(400, "tool_choice naming a function is not supported: it needs constrained decoding; send \"auto\" or \"none\"");
            if (c != "auto" && c != "none") throw BadRequest(400, "tool_choice must be \"auto\" or \"none\"");
            none = c == "none";
        }
        const jmini::Value* tools = given(body, "tools") ? body.get("tools") : nullptr;
        if (!tools) return t;
        if (!tools->isArray()) throw BadRequest(400, "tools must be an array of {\"type\": \"function\", \"function\": {\"name\": a string, ...}}");
        for (size_t i = 0; i < tools->asArray().size(); ++i) {
            const jmini::Value& tool = tools->asArray()[i];
            const std::string at = "tools[" + std::to_string(i) + "]";
            const jmini::Value* type = tool.get("type");
            const jmini::Value* fn = tool.get("function");
            const jmini::Value* name = fn ? fn->get("name") : nullptr;
            const jmini::Value* params = fn ? fn->get("parameters") : nullptr;
            const jmini::Value* about = fn ? fn->get("description") : nullptr;
            if (!type || !type->isString() || type->asString() != "function" || !fn || !fn->isObject() || !name || !name->isString() || name->asString().empty() ||
                (params && !params->isObject()) || (about && !about->isString()))
                throw BadRequest(400, at + " must be {\"type\": \"function\", \"function\": {\"name\": a string, \"description\": a string, \"parameters\": an object}}");
            const jmini::Value* strict = fn->get("strict");
            if (strict && strict->t == jmini::Value::T::Bool && strict->b) throw BadRequest(400, at + ".function.strict is not supported: a strict schema needs constrained decoding");
            if (params) t.typed[name->asString()] = chat::tool_params(*params);
        }
        if (none || tools->asArray().empty()) return t;
        if (format_.tool_format == chat::ToolFormat::none)
            throw BadRequest(400, "tools are not supported by this model's chat template, which writes no tool-call format the server reads");
        try { t.value = chat::template_value(*tools); }
        catch (const chat::TemplateError& e) { throw BadRequest(400, "tools: " + std::string(e.what())); }
        t.on = true;
        return t;
    }

    // A chat request's template variables: its chat_template_kwargs, booleans, numbers, strings or null, none of them one the render itself sets, and its tools where they are given (tools_of).
    std::vector<chat::TemplateVar> template_vars(const jmini::Value& body) const {
        std::vector<chat::TemplateVar> vars;
        Tools tools = tools_of(body);
        if (tools.on) vars.push_back({"tools", std::move(tools.value)});
        const jmini::Value* kwargs = body.get("chat_template_kwargs");
        if (!kwargs || kwargs->t == jmini::Value::T::Null) return vars;
        if (!kwargs->isObject()) throw BadRequest(400, "chat_template_kwargs must be an object");
        static const char* const reserved[] = {"messages", "tools", "documents", "add_generation_prompt", "bos_token", "eos_token"};
        for (const auto& [name, v] : kwargs->obj) {
            if (std::find(std::begin(reserved), std::end(reserved), name) != std::end(reserved))
                throw BadRequest(400, "chat_template_kwargs cannot set " + name);
            if (v.t == jmini::Value::T::Bool) vars.push_back({name, chat::jj::Value::boolean(v.b)});
            else if (v.isString()) vars.push_back({name, chat::jj::Value::str(v.asString())});
            else if (v.t == jmini::Value::T::Null) vars.push_back({name, chat::jj::Value::none()});
            else if (v.isNumber()) {
                try { vars.push_back({name, chat::template_value(v)}); }
                catch (const chat::TemplateError& e) { throw BadRequest(400, "chat_template_kwargs: " + std::string(e.what())); }
            }
            else throw BadRequest(400, "chat_template_kwargs values must be booleans, numbers, strings or null");
        }
        return vars;
    }

    // How much of a request's prompt a follow-up turn begins with, where the model keeps a state a follow-up could fork (chat::stable_prefix for a chat route, a text's whole prompt); a model that keeps none is not asked to render again.
    // The render holds one pad an image, so the prompt is compared with its runs made one pad and the count taken back through them.
    size_t stable_of(const jmini::Value& body, Route route, const std::vector<uint32_t>& prompt) const {
        if (!model_.checkpoint_slots() || (route != Route::chat && route != Route::chat_completions)) return prompt.size();
        return through_runs(prompt, chat::stable_prefix(format_, tok_, messages_of(body), one_pad(prompt), template_vars(body)));
    }

    // Where a chat request's last user message starts, for a model that keeps a state: the text two renders with different last user messages share, counted in the prompt's `ids` lying wholly within it.
    // It is 0 where the message before it is not a reply, a tool's result among them, or the template refuses the render.
    size_t message_start_of(const jmini::Value& body, Route route, const std::string& text, const std::vector<uint32_t>& ids) const {
        if (!model_.checkpoint_slots() || (route != Route::chat && route != Route::chat_completions)) return 0;
        std::vector<chat::Message> messages = messages_of(body);
        size_t last = messages.size();
        while (last > 0 && messages[last - 1].role != "user") --last;
        if (last < 2 || messages[last - 2].role != "assistant") return 0;
        messages.resize(last - 1);
        const std::vector<chat::TemplateVar> vars = template_vars(body);
        std::string with[2];
        try {
            for (int i = 0; i < 2; ++i) {
                std::vector<chat::Message> m = messages;
                m.push_back({"user", i ? "b" : "a", std::nullopt});
                with[i] = format_.render(m, true, vars);
            }
        } catch (const std::exception&) {
            return 0;
        }
        size_t shared = 0;
        while (shared < with[0].size() && shared < with[1].size() && shared < text.size() && with[0][shared] == with[1][shared] && with[0][shared] == text[shared]) ++shared;
        const std::vector<size_t> ends = text_ends(ids);
        return (size_t)(std::upper_bound(ends.begin(), ends.end(), shared) - ends.begin());
    }

    // The ids a chat request's next turn begins with after `text`, its reply as far as it is written, given back as a client of the route sends it (reply_sent_back, docs/SPECULATIVE.md, section 2, Idle re-prefill).
    // None while the reply's reasoning or a call is still being written, which a next turn may drop or would write otherwise; each image takes its run of `rows`, as the prompt does.
    std::vector<uint32_t> next_turn(const jmini::Value& body, Route route, const std::string& prompt, const std::string& text, bool writing,
                                    const std::vector<size_t>& rows) const {
        const bool opened = chat::opens_reasoning(prompt);
        if (writing && (opened || text.find("<think>") != std::string::npos) && text.find("</think>") == std::string::npos) return {};
        std::vector<chat::Message> messages = messages_of(body);
        const Tools tools = tools_of(body);
        bool open = false;
        messages.push_back(reply_sent_back(format_, route == Route::chat_completions, opened, text,
                                           route == Route::chat_completions && tools.on ? format_.tool_format : chat::ToolFormat::none, tools.typed, tools.single, &open));
        if (writing && open) return {};
        return with_runs(chat::stable_prefix(format_, tok_, messages, writing, template_vars(body)), rows);
    }

    // A render the template itself fails, such as its raise_exception on a conversation it does not take, is the request's fault.
    std::string render_messages(const jmini::Value& body) {
        const std::vector<chat::Message> messages = messages_of(body);
        const std::vector<chat::TemplateVar> vars = template_vars(body);
        try { return format_.render(messages, true, vars); }
        catch (const chat::TemplateError& e) { throw BadRequest(400, std::string("the chat template refused the messages: ") + e.what()); }
    }

    // The prompt text of a request on each route.
    std::string prompt_of(const jmini::Value& body, Route route) {
        if (route == Route::chat || route == Route::chat_completions) return render_messages(body);
        const jmini::Value* p = body.get("prompt");
        if (route == Route::completions && p && p->isArray() && p->asArray().size() == 1) p = &p->asArray()[0];
        else if (route == Route::completions && p && p->isArray() && p->asArray().size() > 1) throw BadRequest(400, "prompt must be one string, not an array of several prompts");
        if (!p || !p->isString() || p->asString().empty()) throw BadRequest(400, "prompt must be a non-empty string");
        return p->asString();
    }

    // The fields of a compatible request that llmx cannot honour, refused by name; fields with a harmless default are not read (docs/SERVER.md).
    // Tools are the chat routes' (tools_of); the completions route renders no template to give them to.
    static void refuse_unsupported(const jmini::Value& body, Route route) {
        const auto refuse = [](const char* name, const char* what) { throw BadRequest(400, std::string(name) + " is not supported: " + what); };
        const auto set = [&](const char* key) { return given(body, key) ? body.get(key) : nullptr; };
        const jmini::Value* tools = route == Route::completions ? set("tools") : nullptr;
        if (tools && !(tools->isArray() && tools->asArray().empty())) refuse("tools", "the completions route renders no chat template to give them to");
        const jmini::Value* choice = route == Route::completions ? set("tool_choice") : nullptr;
        if (choice && !(choice->isString() && choice->asString() == "none")) refuse("tool_choice", "the completions route has no tools, so only \"none\" is accepted");
        const jmini::Value* format = set("response_format");
        if (format) {
            const jmini::Value* type = format->isObject() ? format->get("type") : nullptr;
            if (!type || !type->isString() || type->asString() != "text") refuse("response_format", "only {\"type\":\"text\"} is accepted");
        }
        const jmini::Value* bias = set("logit_bias");
        if (bias && !(bias->isObject() && bias->obj.empty())) refuse("logit_bias", "llmx has no logit bias");
        const jmini::Value* best_of = set("best_of");
        if (best_of && !(best_of->isNumber() && best_of->asNumber() == 1)) refuse("best_of", "llmx draws one completion, so only 1 is accepted");
        const jmini::Value* echo = set("echo");
        if (echo && !(echo->t == jmini::Value::T::Bool && !echo->b)) refuse("echo", "llmx does not return the prompt with the completion");
        if (set("suffix")) refuse("suffix", "llmx has no insertion");
        if (set("reasoning_effort")) refuse("reasoning_effort", "llmx has no reasoning budget");
    }

    // The sampling fields, native names first and the compatible routes' synonyms accepted beside them.
    SampleParams params_of(const jmini::Value& body, Route route) {
        SampleParams params;
        const SampleParams defaults;
        const double lo = std::numeric_limits<int>::min(), hi = std::numeric_limits<int>::max();
        // The compatible routes take an absent cap, or -1, as the standard does: no cap, the reply running to the model's end of text or to what the request may hold. The native routes keep the default cap and refuse -1 as any other cap below 1.
        const double cap = integer(body, "max_tokens", lo, hi,
                                   compat(route) ? integer(body, "max_completion_tokens", lo, hi, -1) : defaults.max_tokens);
        params.until_limit = compat(route) && cap == -1;
        params.max_tokens = (int)cap;
        params.temp = setting(body, "temperature", infer::Sampling::temp_range, defaults.temp);
        // Clients send a top_k of -1 for no top-k, so the compatible routes take it as 0, which keeps every token; the native routes refuse it as any other top_k below 0.
        const double top_k = integer(body, "top_k", compat(route) ? -1 : infer::Sampling::top_k_range.lo, infer::Sampling::top_k_range.hi, defaults.top_k);
        params.top_k = top_k == -1 ? 0 : (int)top_k;
        params.top_p = setting(body, "top_p", infer::Sampling::top_p_range, defaults.top_p);
        params.penalty = setting(body, "penalty", infer::Sampling::penalty_range,
                                 compat(route) ? setting(body, "repetition_penalty", infer::Sampling::penalty_range, defaults.penalty) : defaults.penalty);
        // Clients send a seed of -1 for a random one, so the compatible routes take it as no seed, as they take an absent one; the native routes refuse it as any other seed below 0.
        // The largest double below 2^64 is the last one a seed holds.
        const double seed = integer(body, "seed", compat(route) ? -1 : 0, std::nextafter(18446744073709551616.0, 0.0), (double)defaults.seed);
        params.seed = seed == -1 ? defaults.seed : (uint64_t)seed;
        params.presence_penalty = setting(body, "presence_penalty", infer::Sampling::presence_penalty_range, defaults.presence_penalty);
        params.frequency_penalty = setting(body, "frequency_penalty", infer::Sampling::frequency_penalty_range, defaults.frequency_penalty);
        params.min_p = setting(body, "min_p", infer::Sampling::min_p_range, defaults.min_p);
        params.ignore_eos = boolean(body, "ignore_eos", defaults.ignore_eos);
        if (const jmini::Value* stop = body.get("stop")) {
            if (stop->isString()) params.stop.push_back(stop->asString());
            else if (stop->isArray())
                for (const auto& s : stop->asArray())
                    if (s.isString()) params.stop.push_back(s.asString());
        }
        if (compat(route) && integer(body, "n", lo, hi, 1) != 1) throw BadRequest(400, "n must be 1");
        // Log-probabilities in each route's shape: the completions route's `logprobs` is how many of the most likely tokens to list beside each sampled one, and the other routes take `logprobs` true with `top_logprobs` for that count.
        if (route == Route::completions) {
            params.logprobs = given(body, "logprobs");
            if (params.logprobs) params.top_logprobs = (size_t)integer(body, "logprobs", 0, kTopLogprobs, 0);
        } else {
            params.logprobs = given(body, "logprobs") && boolean(body, "logprobs", false);
            if (given(body, "top_logprobs")) params.top_logprobs = (size_t)integer(body, "top_logprobs", 0, kTopLogprobs, 0);
            if (params.top_logprobs && !params.logprobs) throw BadRequest(400, "top_logprobs needs logprobs set to true");
        }
        return params;
    }

    // A sampled token as a reply's logprobs read it: its log-probabilities, its bytes, and the characters of the reply text before the character it starts in.
    struct Sampled {
        Request::Token token;
        std::string bytes;
        size_t offset = 0;
    };
    // A token's text as the logprobs shapes carry it: its bytes when they are whole UTF-8, else "bytes:" and each byte as \xNN, as the compatible APIs spell a token that splits a character.
    static std::string token_text(const std::string& bytes) {
        if (utf8_sanitize(bytes) == bytes) return bytes;
        static constexpr char hex[] = "0123456789abcdef";
        std::string text = "bytes:";
        for (unsigned char c : bytes) {
            text += "\\x";
            text += hex[c >> 4];
            text += hex[c & 0x0f];
        }
        return text;
    }
    // A token in the chat route's shape, {"token", "logprob", "bytes"}, its bytes as numbers so a split character is exact.
    static std::string chat_token(const std::string& bytes, float logprob) {
        std::string list;
        for (unsigned char c : bytes) list += (list.empty() ? "" : ",") + std::to_string(c);
        return "{\"token\":" + jmini::quote(token_text(bytes)) + ",\"logprob\":" + compat_number(logprob) + ",\"bytes\":[" + list + "]";
    }
    // The logprobs of sampled tokens in a compatible route's shape: the chat route's content list beside a null refusal, or the completions route's parallel lists.
    // Their top_logprobs map each listed token's text to its log-probability and hold the sampled token too.
    std::string compat_logprobs(Route route, const Sampled* sampled, size_t n) const {
        if (route == Route::chat_completions) {
            std::string content;
            for (size_t i = 0; i < n; ++i) {
                const Request::Token& t = sampled[i].token;
                std::string top;
                for (const auto& alt : t.top) top += (top.empty() ? "" : ",") + chat_token(tok_.decode({alt.id}), alt.logprob) + "}";
                content += (i ? "," : "") + chat_token(sampled[i].bytes, t.logprob) + ",\"top_logprobs\":[" + top + "]}";
            }
            return "{\"content\":[" + content + "],\"refusal\":null}";
        }
        std::string tokens, values, tops, offsets;
        for (size_t i = 0; i < n; ++i) {
            const Request::Token& t = sampled[i].token;
            const char* sep = i ? "," : "";
            tokens += sep + jmini::quote(token_text(sampled[i].bytes));
            values += sep + compat_number(t.logprob);
            offsets += sep + std::to_string(sampled[i].offset);
            // Two tokens with the same text keep the likelier one's entry, so no key repeats.
            std::vector<std::string> keys;
            std::string top;
            const auto entry = [&](const std::string& key, float logprob) {
                if (std::find(keys.begin(), keys.end(), key) != keys.end()) return;
                top += (keys.empty() ? "" : ",") + jmini::quote(key) + ":" + compat_number(logprob);
                keys.push_back(key);
            };
            bool listed = false;
            for (const auto& alt : t.top) {
                entry(token_text(tok_.decode({alt.id})), alt.logprob);
                listed = listed || alt.id == t.id;
            }
            if (!listed) entry(token_text(sampled[i].bytes), t.logprob);
            tops += sep + std::string("{") + top + "}";
        }
        return "{\"tokens\":[" + tokens + "],\"token_logprobs\":[" + values + "],\"top_logprobs\":[" + tops +
               "],\"text_offset\":[" + offsets + "]}";
    }
    // A native route's most likely tokens at one position, [{"id", "logprob"}].
    static std::string native_top(const Request::Token& t) {
        std::string top;
        for (const auto& alt : t.top)
            top += (top.empty() ? "{\"id\":" : ",{\"id\":") + std::to_string(alt.id) + ",\"logprob\":" + jmini::number(alt.logprob) + "}";
        return "[" + top + "]";
    }

    // A compatible reply's finish_reason: cut by its length, ended after calling tools, or ended.
    static std::string finish_reason(const std::string& finish, bool called) {
        return finish == "length" ? "length" : called ? "tool_calls" : "stop";
    }
    // A reply's call id: the server's start, the reply's number and the call's index, so ids stay unique across restarts.
    std::string call_id(const std::string& reply, size_t index) const {
        return "call_" + std::to_string(started_) + "_" + reply.substr(reply.find('-') + 1) + "_" + std::to_string(index);
    }
    // A whole reply's calls in the compatible chat shape.
    std::string calls_json(const std::string& reply, const std::vector<chat::ToolCall>& calls) const {
        std::string out;
        for (size_t i = 0; i < calls.size(); ++i)
            out += (i ? ",{\"id\":" : "{\"id\":") + jmini::quote(call_id(reply, i)) + ",\"type\":\"function\",\"function\":{\"name\":" + jmini::quote(calls[i].name) +
                   ",\"arguments\":" + jmini::quote(calls[i].arguments.str) + "}}";
        return "[" + out + "]";
    }
    // Streamed pieces of calls as a delta's tool_calls: a call's first piece with its id, type and name, the later ones with their index and arguments alone; empty where there are none.
    std::string deltas_json(const std::string& reply, const std::vector<chat::ToolCallDelta>& pieces) const {
        std::string out;
        for (const chat::ToolCallDelta& d : pieces) {
            out += out.empty() ? "{\"index\":" : ",{\"index\":";
            out += std::to_string(d.index);
            if (!d.name.empty())
                out += ",\"id\":" + jmini::quote(call_id(reply, d.index)) + ",\"type\":\"function\",\"function\":{\"name\":" + jmini::quote(d.name) + ",";
            else out += ",\"function\":{";
            out += "\"arguments\":" + jmini::quote(d.arguments) + "}}";
        }
        return out.empty() ? out : "[" + out + "]";
    }
    std::string usage_json(size_t prompt_tokens, size_t tokens, size_t cached) const {
        return "{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(tokens) +
               ",\"total_tokens\":" + std::to_string(prompt_tokens + tokens) + ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(cached) + "}}";
    }
    // A finished request's speed in the fields clients that display speed read beside the standard usage: prompt tokens prefilled and reused, milliseconds to the first token, and generation after it.
    // `later_ms` is the time of the parts after a shift (docs/SERVER.md, A reply that reaches the context), which generation counts.
    static std::string timings_json(const Request& r, size_t tokens, double later_ms = 0) {
        Request::Timings t = r.timings();
        t.predicted_ms += later_ms;
        const size_t cached = r.reused(), prefilled = r.prompt_tokens() - cached;
        auto num = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.3f", v); return std::string(b); };
        return "{\"cache_n\":" + std::to_string(cached) + ",\"prompt_n\":" + std::to_string(prefilled) +
               ",\"prompt_ms\":" + num(t.prompt_ms) + ",\"prompt_per_second\":" + num(Request::Timings::per_second(prefilled, t.prompt_ms)) +
               ",\"predicted_n\":" + std::to_string(Request::Timings::predicted(tokens)) + ",\"predicted_ms\":" + num(t.predicted_ms) +
               ",\"predicted_per_second\":" + num(t.predicted_per_second(tokens)) + ",\"queued_ms\":" + num(t.queued_ms) +
               ",\"prepared_ms\":" + num(t.prepared_ms) + "}";
    }
    // The head every compatible object and chunk starts with.
    std::string head(const std::string& id, const char* object) const {
        return "{\"id\":" + jmini::quote(id) + ",\"object\":\"" + object + "\",\"created\":" + std::to_string(started_) +
               ",\"model\":" + jmini::quote(cfg_.model_name);
    }
    // One streamed chunk of a compatible route: a chat delta or a text piece, with the finish reason on the last, and the chunk's logprobs when the request asked for them.
    // A chat delta carries `reasoning` as reasoning_content beside its content when a reply's reasoning is split off, and `calls`, pieces of tool calls (deltas_json), where it has any.
    std::string chunk(Route route, const std::string& id, const std::string& piece, bool first, const std::string* finish, const std::string& logprobs = "",
                      const std::string* reasoning = nullptr, const std::string& calls = "") const {
        const std::string fr = finish ? jmini::quote(*finish) : "null";
        const std::string lp = logprobs.empty() ? "" : ",\"logprobs\":" + logprobs;
        if (route == Route::chat_completions) {
            const std::string thought = reasoning ? "\"reasoning_content\":" + jmini::quote(*reasoning) + "," : "";
            const std::string called = calls.empty() ? "" : ",\"tool_calls\":" + calls;
            std::string delta = first ? "{\"role\":\"assistant\"," + thought + "\"content\":" + jmini::quote(piece) + called + "}"
                              : finish ? "{}" : "{" + thought + "\"content\":" + jmini::quote(piece) + called + "}";
            return head(id, "chat.completion.chunk") + ",\"choices\":[{\"index\":0,\"delta\":" + delta + lp +
                   ",\"finish_reason\":" + fr + "}]}";
        }
        return head(id, "text_completion") + ",\"choices\":[{\"index\":0,\"text\":" + jmini::quote(piece) + lp +
               ",\"finish_reason\":" + fr + "}]}";
    }
    // The same chunk with a request's timings, for the one that carries the finish reason, `reason` as finish_reason gives it.
    std::string last_chunk(Route route, const std::string& id, const std::string& reason, const Request& r, size_t tokens,
                           const std::string& logprobs, double later_ms) const {
        std::string c = chunk(route, id, "", false, &reason, logprobs);
        c.pop_back();
        return c + ",\"timings\":" + timings_json(r, tokens, later_ms) + "}";
    }

    // A prompt's ids, a text the tokenizer cannot encode refused with 400 on every route that reads one.
    std::vector<uint32_t> encode(const std::string& text) const {
        try { return tok_.encode(text); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }
    }

    // The ids a prompt of `text` reads, or of `messages` rendered as the chat routes render them.
    // `add_special` is not read, since the generating routes add no start or end token to a prompt and there is none to add or leave out.
    void tokenize(http::Connection& c, const http::Request& req) {
        const jmini::Value body = body_of(req);
        const bool chat = body.get("messages") != nullptr;
        const jmini::Value* text = body.get("text");
        if (chat && text) throw BadRequest(400, "give text or messages, not both");
        if (!chat && (!text || !text->isString())) throw BadRequest(400, "text must be a string");
        std::vector<uint32_t> ids = encode(chat ? render_messages(body) : text->asString());
        // An image counts the rows its size gives, without being encoded.
        const std::vector<std::string> urls = image_urls(body);
        if (!urls.empty()) {
            if (!cfg_.vision) throw BadRequest(400, "this server reads no images; start it with --mmproj");
            std::vector<size_t> rows;
            try {
                for (const std::string& url : urls) rows.push_back(infer::image_tokens(*cfg_.vision, image_bytes(url)));
                ids = infer::expand(*cfg_.vision, ids, rows).ids;
            } catch (const BadRequest&) {
                throw;
            } catch (const std::exception& e) {
                throw BadRequest(400, std::string("an image was refused: ") + e.what());
            }
        }
        c.respond(200, "application/json", "{\"tokens\":" + ids_json(ids) + ",\"count\":" + std::to_string(ids.size()) + "}");
    }

    // The text of token ids with the repair a reply's text gets, so the ids of a whole reply give back its text.
    void detokenize(http::Connection& c, const http::Request& req) {
        const jmini::Value body = body_of(req);
        const jmini::Value* tokens = body.get("tokens");
        if (!tokens || !tokens->isArray()) throw BadRequest(400, "tokens must be an array of token ids");
        std::vector<uint32_t> ids;
        for (const auto& t : tokens->asArray()) ids.push_back((uint32_t)integer_value(t, "each token id", 0, (double)tok_.vocab.size() - 1));
        c.respond(200, "application/json", "{\"text\":" + jmini::quote(utf8_sanitize(tok_.decode(ids))) + "}");
    }

    // What a cut dropped of a prompt, and of a history as its reply is written, for the reply's context object and the request's line.
    struct Fitted {
        size_t messages = 0, tokens = 0, renders = 0;   // renders: those that placed the messages' ends, on a chat route
        bool cut = false, at_marker = false;
        size_t replies = 0;       // the shifts while the reply was written (docs/SERVER.md, A reply that reaches the context)
        size_t prompt_rows = 0;   // after a cut, the kept history's rows before the reply's kept rows; on a text route all of them
        std::vector<uint32_t> dropped;   // the uncut prompt through the first row the cut dropped, whose holders the scheduler supersedes (Scheduler::submit)
        // A shift of the same reply's history added: what every cut dropped, at a marker only where each was.
        void add(const Fitted& later) {
            at_marker = (cut ? at_marker : true) && later.at_marker;
            cut = true;
            messages += later.messages;
            tokens += later.tokens;
            ++replies;
        }
    };

    // Where each message of a conversation ends in the prompt's `ids`: the end tokens, where they number the messages and the template's own, else each part asked about rendered as text.
    // A rendered part takes the ids whose text lies within it, and one the template refuses is placed by the end tokens (MessageEnds, docs/SERVER.md, A prompt larger than the context).
    MessageEnds message_ends(const std::vector<chat::Message>& messages, const std::vector<chat::TemplateVar>& vars, const std::vector<uint32_t>& ids) const {
        std::vector<size_t> points = cut_points(ids.data(), ids.size(), [this](uint32_t id) { return tok_.is_eos(id); });
        const size_t own = chat::own_end_tokens(format_, tok_, messages, vars);
        if (points.size() == messages.size() + own) return MessageEnds(std::vector<size_t>(points.begin() + (std::ptrdiff_t)own, points.end()));
        auto bytes = std::make_shared<std::vector<size_t>>();
        return MessageEnds(messages.size(), ids.size(), std::move(points), [this, &messages, &vars, &ids, bytes](size_t k) -> std::optional<size_t> {
            std::string text;
            try { text = format_.render(std::vector<chat::Message>(messages.begin(), messages.begin() + (std::ptrdiff_t)k), false, vars); }
            catch (const chat::TemplateError&) { return std::nullopt; }
            if (bytes->empty()) *bytes = text_ends(ids);
            // A part ends with its last text, as its end token ends it, not with the whitespace a template writes after the token.
            size_t len = text.size();
            while (len > 0 && (text[len - 1] == '\n' || text[len - 1] == ' ')) --len;
            return (size_t)(std::upper_bound(bytes->begin(), bytes->end(), len) - bytes->begin());
        });
    }

    // The turns that reused little and the last one's numbers (Cache::note_miss), its message counted from 1 and null on a route without messages.
    static std::string misses_json(const Scheduler::Stats& s) {
        const auto n = [](uint64_t v) { return std::to_string(v); };
        const ReuseMiss& m = s.last_miss;
        const std::string last = !s.reuse_misses ? "null" : "{\"prompt\":" + n(m.prompt) + ",\"reused\":" + n(m.reused) + ",\"agreed\":" + n(m.agreed) + ",\"limit\":" + jmini::quote(limit_name(m.limit)) +
            ",\"at\":" + n(m.at) + ",\"message\":" + (m.messages ? n(m.message + 1) : "null") + ",\"messages\":" + n(m.messages) + ",\"role\":" + (m.messages ? jmini::quote(m.role) : "null") + "}";
        return ",\"misses\":{\"since_start\":" + n(s.reuse_misses) + ",\"last\":" + last + "}";
    }

    // Where each message of a chat prompt ends in `ids` and its role, from the end tokens alone with no render, for the note of a turn that reuses little (Cache::note_miss).
    // A template's own end tokens come before the messages', so the last ones are the messages'; with fewer end tokens than messages, as where turns merge, nothing.
    MessageSpans message_spans(const std::vector<chat::Message>& messages, const std::vector<uint32_t>& ids) const {
        const std::vector<size_t> points = cut_points(ids.data(), ids.size(), [this](uint32_t id) { return tok_.is_eos(id); });
        MessageSpans s;
        if (messages.empty() || points.size() < messages.size()) return s;
        s.ends.assign(points.end() - (std::ptrdiff_t)messages.size(), points.end());
        for (const chat::Message& m : messages) s.roles.push_back(m.role);
        return s;
    }

    // A prompt that does not fit what a request may hold is refused with its numbers, or, with --context-overflow shift, cut at a cut point in steps of half of that (docs/SERVER.md, A prompt larger than the context).
    // With `reply`, the rows a reply holds after its prompt as it reaches the limit, the history is cut alike (docs/SERVER.md, A reply that reaches the context); `body`, `prompt` and `ids` are the cut request's afterwards.
    Fitted fit(jmini::Value& body, Route route, const SampleParams& params, std::string& prompt, std::vector<uint32_t>& ids, const RequestImages& images,
               const std::vector<uint32_t>& reply = {}) {
        // The rows the prompt may take beside its reply, the scheduler's rule, and the step a cut drops in.
        const size_t limit = sched_.token_limit(), n = ids.size() + reply.size(), room = sched_.prompt_room(params), step = limit / kCutTo;
        const auto fits = [&](size_t rows) { return rows <= room; };
        if (fits(n)) return {};
        const auto refuse = [&](const std::string& why) { throw BadRequest(413, sched_.too_long(n, params) + "; " + why); };
        if (!cfg_.context_shift)
            refuse(std::string(params.until_limit ? "shorten it, send max_tokens" : "shorten it") + ", or start the server with --context-overflow shift, which drops the oldest turns");
        Fitted f;
        f.cut = true;
        std::vector<uint32_t> uncut = ids;
        uncut.insert(uncut.end(), reply.begin(), reply.end());
        bool tail = false;
        if (route == Route::chat || route == Route::chat_completions) {
            const std::vector<chat::Message> messages = messages_of(body);
            const std::vector<chat::TemplateVar> vars = template_vars(body);
            MessageEnds ends = message_ends(messages, vars, ids);
            std::vector<char> starts;
            for (const chat::Message& m : messages) starts.push_back(m.role == "user");
            const size_t lead = messages[0].role == "system" ? 1 : 0, k = first_kept(ends, starts, lead, n, room, step);
            // Where no whole turn serves, the newest turn's oldest tool exchanges go, each an assistant message with the results after it (cut_inside).
            TurnCut inside{k, k};
            if (k == messages.size()) {
                std::vector<char> exchanges;
                for (const chat::Message& m : messages) exchanges.push_back(m.role == "assistant");
                inside = cut_inside(ends, starts, exchanges, lead, n, room, step);
                if (!reply.empty() && inside.refused != TurnCut::Refused::none && inside.user >= lead && starts[inside.user]) {
                    // A reply too long beside its turn: every message after the newest user message goes, and the reply's oldest rows below.
                    inside.from = messages.size();
                    tail = true;
                } else if (inside.refused == TurnCut::Refused::head) {
                    refuse(std::string(lead ? "its system message and " : "its ") + "newest user message alone do not fit, so no cut helps");
                } else if (inside.refused == TurnCut::Refused::exchange) {
                    refuse("its newest user message and newest tool exchange alone do not fit, and a message is not cut inside");
                }
            }
            f.renders = ends.renders() + 1;
            for (auto& member : body.obj)
                if (member.first == "messages") {
                    auto& arr = member.second.arr;
                    arr.erase(arr.begin() + (std::ptrdiff_t)inside.user + (inside.from > inside.user ? 1 : 0), arr.begin() + (std::ptrdiff_t)inside.from);
                    arr.erase(arr.begin() + (std::ptrdiff_t)lead, arr.begin() + (std::ptrdiff_t)inside.user);
                }
            prompt = prompt_of(body, route);
            ids = prompt_ids(body, prompt, images);
            f.messages = (inside.user - lead) + (inside.from - inside.user - (inside.from > inside.user ? 1 : 0));
            f.at_marker = !tail;
            // The reply's oldest rows go in whole steps from its start, the fewest after which the history fits.
            const size_t over = ids.size() + reply.size() > room ? ids.size() + reply.size() - room : 0, from = tail ? std::min(reply.size(), (over + step - 1) / step * step) : 0;
            f.prompt_rows = ids.size();
            ids.insert(ids.end(), reply.begin() + (std::ptrdiff_t)from, reply.end());
        } else {
            ids = uncut;
            const RowCut c = cut_rows(cut_points(ids.data(), n, [this](uint32_t id) { return tok_.is_eos(id); }), n, room, step);
            if (c.from >= n) refuse("its newest rows alone do not fit, so no cut helps");
            ids.erase(ids.begin() + (std::ptrdiff_t)c.lead, ids.begin() + (std::ptrdiff_t)c.from);
            f.at_marker = c.at_marker;
            f.prompt_rows = ids.size();
        }
        f.tokens = n - ids.size();
        size_t same = 0;
        while (same < ids.size() && same < uncut.size() && ids[same] == uncut[same]) ++same;
        f.dropped.assign(uncut.begin(), uncut.begin() + (std::ptrdiff_t)std::min(same + 1, uncut.size()));
        if (!fits(ids.size())) refuse("cut to " + std::to_string(ids.size()) + " it still does not fit with what it asks for");
        ++shifts_;
        shifted_tokens_ += f.tokens;
        const std::string messages = f.messages ? ", " + std::to_string(f.messages) + " messages" : std::string();
        const std::string renders = f.renders ? ", its messages' ends found in " + std::to_string(f.renders) + " render" + (f.renders == 1 ? "" : "s") : std::string();
        std::fprintf(stderr, "server: a %s of %zu tokens cut to %zu to fit %zu, %zu tokens dropped%s%s%s\n", reply.empty() ? "prompt" : "reply's history", n, ids.size(), limit, f.tokens,
                     messages.c_str(), renders.c_str(), f.at_marker ? "" : ", not at a turn's end");
        return f;
    }

    // A reply's context object: the rows the request holds against what it may, `held`, and what the cuts dropped of its prompt and, after a shift, of its history.
    std::string context_json(size_t held, const Fitted& f) const {
        std::string c = "{\"tokens\":" + std::to_string(held) + ",\"limit\":" + std::to_string(sched_.token_limit());
        if (f.cut) {
            if (f.messages) c += ",\"dropped_messages\":" + std::to_string(f.messages);
            c += ",\"dropped_tokens\":" + std::to_string(f.tokens) + ",\"at_marker\":" + (f.at_marker ? "true" : "false");
            if (f.replies) c += ",\"reply_shifts\":" + std::to_string(f.replies);
        }
        return c + "}";
    }

    // The images a prompt holds after any cut, which the scheduler encodes from their resized pixels wherever a sequence reads their rows, a recompute among them.
    Scheduler::Images images_given(const jmini::Value& body, const std::string& prompt, const RequestImages& images) const {
        Scheduler::Images given;
        if (image_rows_of(body, images).empty()) return given;
        (void)prompt_ids(body, prompt, images, &given.runs);
        std::vector<std::shared_ptr<const image::Raster>> resized;
        for (const std::string& url : image_urls(body)) resized.push_back(images.prepared.at(url).resized);
        given.encode = [vision = cfg_.vision, resized](size_t i) {
            auto e = std::make_shared<infer::ImageEncode>(*vision, *resized.at(i));
            auto ms = std::make_shared<double>(0.0);
            return ImageSteps{[vision, e, ms]() -> std::shared_ptr<const std::vector<float>> {
                const auto t0 = std::chrono::steady_clock::now();
                const bool done = e->step();
                *ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (!done) return nullptr;
                auto rows = e->rows();
                std::fprintf(stderr, "server: an image of %zu rows encoded in %zu pieces, %.1f ms of them\n", rows->size() / vision->encoder->plan().out_width, e->pieces(), *ms);
                return rows;
            }, e->pieces()};
        };
        return given;
    }

    void generate(http::Connection& c, const http::Request& req, Route route) {
        // The time to first token counts from here, so the render, the tokenizer and a cut are in it.
        const Request::Clock::time_point received = Request::Clock::now();
        jmini::Value body = body_of(req);
        if (compat(route)) refuse_unsupported(body, route);
        std::string prompt = prompt_of(body, route);
        const SampleParams params = params_of(body, route);
        const bool stream = flag(body, "stream");
        const jmini::Value* so = body.get("stream_options");
        const bool include_usage = so && so->isObject() && flag(*so, "include_usage");

        std::shared_ptr<Request> r;
        const RequestImages images = images_of(body, route);
        std::vector<uint32_t> ids = prompt_ids(body, prompt, images);
        Fitted fitted = fit(body, route, params, prompt, ids, images);
        // What the scheduler takes beside a history's ids, from the body as a cut left it: the prefix a follow-up turn begins with, where its last user message starts, its images and where its messages end.
        struct Parts {
            size_t stable = 0, start = 0;
            Scheduler::Images given;
            MessageSpans spans;
        };
        const auto parts_of = [&](const std::vector<uint32_t>& history) {
            Parts p;
            p.stable = stable_of(body, route, history);
            p.start = message_start_of(body, route, prompt, history);
            p.given = images_given(body, prompt, images);
            if (route == Route::chat || route == Route::chat_completions) p.spans = message_spans(messages_of(body), history);
            return p;
        };
        const std::vector<size_t> image_rows = image_rows_of(body, images);
        // The prompt's rows of the history the request holds, and where in the reply its kept rows begin: all of it until a shift.
        std::vector<uint32_t> base = ids;
        size_t from = 0;
        Parts parts = parts_of(ids);
        try { r = sched_.submit(std::move(ids), params, parts.stable, parts.start, received, fitted.dropped, std::move(parts.given), std::move(parts.spans)); }
        catch (const TooLong& e) { throw BadRequest(413, e.what()); }
        catch (const QueueFull& e) { throw BadRequest(503, e.what()); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }
        // The first part, whose prompt and reuse the reply reports, the part's first token in the reply, and the time of the parts after it.
        const std::shared_ptr<Request> first_part = r;
        size_t part_start = 0;
        double later_ms = 0;
        const auto spent = [](const Request& q) {
            const Request::Timings t = q.timings();
            return t.prepared_ms + t.queued_ms + t.prompt_ms + t.predicted_ms;
        };

        const std::string id = (route == Route::chat_completions ? "chatcmpl-" : "cmpl-") + std::to_string(next_id_.fetch_add(1));
        // A chat reply gives its reasoning as reasoning_content, apart from its content, as clients show a reasoning model's: inside the <think> its template opened or it opened itself.
        const bool split = route == Route::chat_completions;
        chat::ReplySplit splitter(chat::opens_reasoning(prompt));
        // With tools, its content's tool calls apart from its text, as calls in the compatible shape.
        const Tools tools = route == Route::chat || split ? tools_of(body) : Tools{};
        chat::ToolCallSplit caller(split && tools.on ? format_.tool_format : chat::ToolFormat::none, tools.typed, tools.single);
        // With one call a reply, on either chat route, the request is cancelled at the first call's close and what it draws after is left out.
        const bool single = tools.on && tools.single;
        chat::ReplySplit watch_reasoning(chat::opens_reasoning(prompt));
        chat::ToolCallSplit watch(single ? format_.tool_format : chat::ToolFormat::none, tools.typed, true);
        bool cut = false;

        // Drain the channel; a write that fails means the client went away, so cancel the request and stop.
        // Between writes the socket is looked at every kProbe, token or not, since a request that is queued, prefilling or building a whole reply writes nothing that could fail.
        std::vector<uint32_t> gen;
        std::string text, pending;
        bool first = true;
        // The compatible routes leave a stop text out of the reply, holding back the end of the text while it could still become one.
        const bool trims = compat(route) && !params.stop.empty();
        StopTrim trim(params.stop);
        // With logprobs, a whole reply's tokens, and the characters of the text so far, where the next token's offset is.
        std::vector<Sampled> sampled;
        size_t chars = 0;
        // What a compatible chunk that carries no token holds for its logprobs when the request asked for them.
        const std::string no_logprobs = params.logprobs ? "null" : "";
        // A chat reply's next turn goes to the scheduler as it is written and, once, as it has ended, before the reply's last words go out, so its job knows them before a client can answer; nothing where the reply did not end as a reply.
        bool following = (route == Route::chat || route == Route::chat_completions) && sched_.follows();
        const auto close_follow = [&](bool ended) {
            if (following) sched_.follow(r, ended ? next_turn(body, route, prompt, text, false, image_rows) : std::vector<uint32_t>{}, true);
            following = false;
        };
        // An uncapped reply that reached what a request may hold goes on under --context-overflow shift (docs/SERVER.md, A reply that reaches the context): its history is cut as a prompt is, and the reply continues on what is kept as one reply.
        // A history no cut fits ends the reply there, with length.
        const auto shift = [&]() {
            if (cut || !cfg_.context_shift || !params.until_limit || r->finish() != "length") return false;
            const Request::Clock::time_point at = Request::Clock::now();
            std::vector<uint32_t> history = base, reply(gen.begin() + (std::ptrdiff_t)from, gen.end());
            Fitted f;
            try { f = fit(body, route, params, prompt, history, images, reply); }
            catch (const BadRequest& e) {
                std::fprintf(stderr, "server: a reply of %zu tokens ends at the limit, since %s\n", gen.size(), e.what());
                return false;
            }
            if (!f.cut) return false;
            // A cut conversation's next turn is cut again, so no job reads it ahead.
            close_follow(false);
            fitted.add(f);
            base.assign(history.begin(), history.begin() + (std::ptrdiff_t)f.prompt_rows);
            from = gen.size() - (history.size() - f.prompt_rows);
            if (r != first_part) later_ms += spent(*r);
            Parts p = parts_of(history);
            r = sched_.carry_on(*r, std::move(history), p.stable, p.start, at, std::move(f.dropped), std::move(p.given), std::move(p.spans));
            part_start = gen.size();
            return true;
        };
        try {
            if (stream) c.begin_stream(200, "text/event-stream");
            Request::Token tok;
            Request::Clock::time_point probe = Request::Clock::now() + kProbe;
            for (;;) {
                const Request::Next got = r->next(tok, probe);
                if (got == Request::Next::end) {
                    if (shift()) continue;
                    break;
                }
                if (Request::Clock::now() >= probe) {
                    if (c.peer_closed()) throw http::ClientGone("http: the client closed the connection");
                    probe = Request::Clock::now() + kProbe;
                }
                if (got == Request::Next::timeout || cut) continue;
                gen.push_back(tok.id);
                std::string bytes = tok_.decode({tok.id});
                pending += bytes;
                const size_t whole = utf8_complete(pending);
                const std::string piece = utf8_sanitize(pending.substr(0, whole));
                pending.erase(0, whole);
                const std::string out = trims ? trim.feed(piece) : piece;
                text += out;
                if (single && (watch.feed(watch_reasoning.feed(out).content), watch.done())) {
                    r->cancel();
                    cut = true;
                }
                if (following && gen.size() % kFollowEvery == 0) sched_.follow(r, next_turn(body, route, prompt, text, true, image_rows), false);
                Sampled s;
                if (params.logprobs) {
                    s = Sampled{std::move(tok), std::move(bytes), chars};
                    for (unsigned char b : piece) chars += (b & 0xC0) != 0x80;
                }
                if (!stream) {
                    if (params.logprobs) sampled.push_back(std::move(s));
                    continue;
                }
                if (compat(route)) {
                    const std::string lp = params.logprobs ? compat_logprobs(route, &s, 1) : "";
                    // A token whose text is all held back sends a chunk only for its logprobs or to open the stream.
                    if (trims && out.empty() && lp.empty() && !first) continue;
                    if (split) {
                        chat::ReplySplit::Parts parts = splitter.feed(out);
                        std::string calls;
                        if (tools.on) {
                            const chat::ToolCallSplit::Parts t = caller.feed(parts.content);
                            parts.content = t.content;
                            calls = deltas_json(id, t.calls);
                            if (parts.content.empty() && parts.reasoning.empty() && calls.empty() && lp.empty() && !first) continue;
                        }
                        c.write_chunk("data: " + chunk(route, id, parts.content, first, nullptr, lp, parts.reasoning.empty() ? nullptr : &parts.reasoning, calls) + "\n\n");
                    } else {
                        c.write_chunk("data: " + chunk(route, id, out, first, nullptr, lp) + "\n\n");
                    }
                } else {
                    std::string lp;
                    if (params.logprobs) lp = ",\"logprob\":" + jmini::number(s.token.logprob) + (params.top_logprobs ? ",\"top_logprobs\":" + native_top(s.token) : "");
                    c.write_chunk("data: {\"id\":" + std::to_string(gen.back()) + ",\"text\":" + jmini::quote(piece) + lp + "}\n\n");
                }
                first = false;
            }
            // Whatever is left never completed a character.
            pending = utf8_sanitize(pending);
            const std::string rest = trims ? trim.feed(pending) + trim.finish() : pending;
            text += rest;
            const std::string finish = cut ? "stop" : r->finish();
            if (finish == "error") {
                if (!stream) throw std::runtime_error(r->error());
                // The stream's head went out as 200, so the failure is its last event.
                c.write_chunk("data: " + error_json(r->error(), compat(route)) + "\n\n");
                c.end_stream();
                close_follow(false);
                return;
            }
            close_follow(finish != "cancel");
            const size_t prompt_tokens = first_part->prompt_tokens(), held = r->prompt_tokens() + (gen.size() - part_start);
            if (r != first_part) later_ms += spent(*r);
            if (stream && compat(route)) {
                if (split) {
                    chat::ReplySplit::Parts parts = splitter.feed(rest);
                    const chat::ReplySplit::Parts last = splitter.finish();
                    parts.reasoning += last.reasoning;
                    parts.content += last.content;
                    std::string calls;
                    if (tools.on) {
                        chat::ToolCallSplit::Parts t = caller.feed(parts.content);
                        const chat::ToolCallSplit::Parts end = caller.finish();
                        t.content += end.content;
                        t.calls.insert(t.calls.end(), end.calls.begin(), end.calls.end());
                        parts.content = t.content;
                        calls = deltas_json(id, t.calls);
                    }
                    if (!parts.reasoning.empty() || !parts.content.empty() || !calls.empty() || first)
                        c.write_chunk("data: " + chunk(route, id, parts.content, first, nullptr, no_logprobs, parts.reasoning.empty() ? nullptr : &parts.reasoning, calls) + "\n\n");
                } else if (!rest.empty() || first) {
                    c.write_chunk("data: " + chunk(route, id, rest, first, nullptr, no_logprobs) + "\n\n");
                }
                c.write_chunk("data: " + last_chunk(route, id, finish_reason(finish, caller.calls() > 0), *first_part, gen.size(), no_logprobs, later_ms) + "\n\n");
                if (include_usage)
                    c.write_chunk("data: " + head(id, route == Route::chat_completions ? "chat.completion.chunk" : "text_completion") +
                                  ",\"choices\":[],\"usage\":" + usage_json(prompt_tokens, gen.size(), first_part->reused()) + ",\"context\":" + context_json(held, fitted) + "}\n\n");
                c.write_chunk("data: [DONE]\n\n");
                c.end_stream();
            } else if (stream) {
                if (!pending.empty()) c.write_chunk("data: {\"text\":" + jmini::quote(pending) + "}\n\n");
                c.write_chunk("data: {\"done\":true,\"finish\":" + jmini::quote(finish) +
                              ",\"tokens\":" + std::to_string(gen.size()) + ",\"context\":" + context_json(held, fitted) + "}\n\ndata: [DONE]\n\n");
                c.end_stream();
            } else if (compat(route)) {
                const std::string lp = params.logprobs ? ",\"logprobs\":" + compat_logprobs(route, sampled.data(), sampled.size()) : "";
                std::string message = "\"content\":" + jmini::quote(text);
                if (split) {
                    const chat::ReplySplit::Parts parts = splitter.feed(text);
                    const chat::ReplySplit::Parts last = splitter.finish();
                    std::string content = parts.content + last.content;
                    std::vector<chat::ToolCall> calls;
                    if (tools.on) {
                        const chat::ToolCallSplit::Parts a = caller.feed(content), b = caller.finish();
                        content = a.content + b.content;
                        chat::gather_calls(calls, a.calls);
                        chat::gather_calls(calls, b.calls);
                    }
                    // A reply of calls alone has a null content, as the compatible API gives it.
                    message = (splitter.reasoned() ? "\"reasoning_content\":" + jmini::quote(parts.reasoning + last.reasoning) + "," : std::string()) +
                              "\"content\":" + (calls.empty() || !content.empty() ? jmini::quote(content) : std::string("null")) +
                              (calls.empty() ? std::string() : ",\"tool_calls\":" + calls_json(id, calls));
                }
                const std::string choice = route == Route::chat_completions
                    ? "{\"index\":0,\"message\":{\"role\":\"assistant\"," + message + "}"
                    : "{\"index\":0,\"text\":" + jmini::quote(text);
                c.respond(200, "application/json",
                          head(id, route == Route::chat_completions ? "chat.completion" : "text_completion") + ",\"choices\":[" + choice + lp +
                          ",\"finish_reason\":" + jmini::quote(finish_reason(finish, caller.calls() > 0)) + "}],\"usage\":" +
                          usage_json(prompt_tokens, gen.size(), first_part->reused()) + ",\"context\":" + context_json(held, fitted) + ",\"timings\":" + timings_json(*first_part, gen.size(), later_ms) + "}");
            } else {
                // With logprobs, a list beside the ids, and with top_logprobs a list of each position's most likely tokens.
                std::string lp;
                if (params.logprobs) {
                    std::string values, tops;
                    for (size_t i = 0; i < sampled.size(); ++i) {
                        values += (i ? "," : "") + jmini::number(sampled[i].token.logprob);
                        tops += (i ? "," : "") + native_top(sampled[i].token);
                    }
                    lp = ",\"logprobs\":[" + values + "]" + (params.top_logprobs ? ",\"top_logprobs\":[" + tops + "]" : "");
                }
                c.respond(200, "application/json",
                          "{\"text\":" + jmini::quote(text) + ",\"ids\":" + ids_json(gen) + lp +
                          ",\"finish\":" + jmini::quote(finish) + ",\"prompt_tokens\":" +
                          std::to_string(prompt_tokens) + ",\"reused_tokens\":" + std::to_string(first_part->reused()) +
                          ",\"tokens\":" + std::to_string(gen.size()) + ",\"context\":" + context_json(held, fitted) + "}");
            }
        } catch (...) {
            r->cancel();
            Request::Token drop;
            while (r->next(drop, Request::Clock::now() + kProbe) != Request::Next::end) {}
            close_follow(false);
            throw;
        }
    }

    infer::Model& model_;
    const bpe::Tokenizer& tok_;
    const chat::ChatFormat format_;
    Scheduler& sched_;
    Config cfg_;
    const std::string cached_topology_json_;
    const Startup* startup_;   // the start's record, which health keeps, where the server has one
    const int64_t started_;
    std::atomic<uint64_t> shifts_{0}, shifted_tokens_{0};   // since start, the requests whose prompt a shift cut, and the tokens those cuts dropped
    std::atomic<uint64_t> next_id_{1};
};

namespace detail {
// The listener SIGTERM and SIGINT, or a console's Ctrl-C, Ctrl-Break and close, stop while serve runs.
inline std::atomic<http::Listener*>& signalled_listener() {
    static std::atomic<http::Listener*> l{nullptr};
    return l;
}
#if defined(_WIN32)
inline BOOL WINAPI on_console(DWORD) {
    static std::atomic<int> seen{0};
    http::Listener* l = signalled_listener().load();
    if (!l || seen.fetch_add(1) > 0) return FALSE;
    l->request_close();
    return TRUE;
}
#else
inline void on_signal(int) {
    if (http::Listener* l = signalled_listener().load()) l->request_close();
}
#endif
// While it lives, the first such signal stops the listener, as closing it does, so the server exits cleanly; the next ends the process.
struct StopOnSignal {
    explicit StopOnSignal(http::Listener& l) {
        signalled_listener().store(&l);
#if defined(_WIN32)
        SetConsoleCtrlHandler(on_console, TRUE);
#else
        struct sigaction sa {};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESETHAND;
        sigaction(SIGTERM, &sa, &term_);
        sigaction(SIGINT, &sa, &int_);
#endif
    }
    ~StopOnSignal() {
#if defined(_WIN32)
        SetConsoleCtrlHandler(on_console, FALSE);
#else
        sigaction(SIGTERM, &term_, nullptr);
        sigaction(SIGINT, &int_, nullptr);
#endif
        signalled_listener().store(nullptr);
    }
    StopOnSignal(const StopOnSignal&) = delete;
    StopOnSignal& operator=(const StopOnSignal&) = delete;
#if !defined(_WIN32)
    struct sigaction term_ {}, int_ {};
#endif
};
} // namespace detail

// The server's accept loop from the moment it listens, on a thread of its own: until the routes are published, health answers 503 with the start's record and every other route 503.
// It ends when the listener closes; destroyed, it closes the listener and waits for its connections.
class Front {
public:
    Front(http::Listener& listener, Startup& startup) : listener_(listener), startup_(startup) {
        thread_ = std::thread([this] { loop(); });
    }
    ~Front() {
        listener_.close();
        wait();
        drain();
    }
    Front(const Front&) = delete;
    Front& operator=(const Front&) = delete;

    http::Listener& listener() { return listener_; }
    Startup& startup() { return startup_; }
    // From now on each connection goes to `api`, which outlives the connections (drain).
    void publish(Api* api) { api_.store(api); }
    // Until the listener closes.
    void wait() {
        if (thread_.joinable()) thread_.join();
    }
    // Until every connection's thread has ended.
    void drain() {
        while (open_.load() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

private:
    void loop() {
        for (;;) {
            http::Connection c = listener_.accept();
            if (!c.open()) break;
            ++open_;
            std::thread([this](http::Connection conn) {
                try {
                    if (Api* api = api_.load()) api->handle(conn);
                    else starting(conn);
                } catch (...) {
                }
                --open_;
            }, std::move(c)).detach();
        }
    }
    // A request while the server starts: health gives the record with 503, so a readiness check on 200 keeps its meaning, liveness answers as ever, and every other route is refused with 503 in its clients' shape.
    void starting(http::Connection& c) {
        http::Request req;
        int status = 0;
        const http::Limits limits;
        const bool read = c.read_request(req, limits, status);
        const std::optional<Api::Route> route = Api::route_of(req.path);
        const bool shape = route && Api::compat(*route);
        if (!read) {
            if (status) c.respond(status, "application/json", error_json(http::reason(status), shape));
            return;
        }
        if (req.method == "GET" && req.path == "/v1/health") return c.respond(503, "application/json", startup_.starting());
        if (req.method == "GET" && req.path == "/v1/live") return c.respond(200, "application/json", "{\"status\":\"ok\"}");
        c.respond(503, "application/json", error_json("the server is starting", shape));
    }

    http::Listener& listener_;
    Startup& startup_;
    std::atomic<Api*> api_{nullptr};
    std::atomic<int> open_{0};
    std::thread thread_;
};

// Serve the routes on `front` until its listener is closed, or a signal stops it (detail::StopOnSignal): the scheduler on its own thread.
inline void serve(infer::Model& model, const bpe::Tokenizer& tok, const chat::ChatFormat& format,
                  const Config& cfg, Front& front) {
    Scheduler sched(model, tok, cfg.max_seqs, cfg.max_queue, cfg.passes, cfg.timing,
                    cfg.host_cache_bytes.value_or(default_host_cache(model, cfg.max_seqs, Scheduler::request_limit(model, cfg.ctx_size))), cfg.proposer, cfg.draft_max, true, cfg.disk,
                    cfg.ctx_size);
    const Scheduler::Stats started = sched.stats();
    std::fprintf(stderr, "server: up to %zu pass%s in flight over %zu stage%s, %zu sampling thread%s beside the scheduler's\n", started.passes,
                 started.passes == 1 ? "" : "es", model.stage_count(), model.stage_count() == 1 ? "" : "s", started.samplers,
                 started.samplers == 1 ? "" : "s");
    Api api(model, tok, format, sched, cfg, &front.startup());
    std::thread runner([&] { sched.run(); });
    const detail::StopOnSignal stop_on_signal(front.listener());
    front.startup().ready();
    front.publish(&api);
    front.wait();
    sched.stop();
    runner.join();
    // Connection threads still draining end with their requests cancelled.
    front.drain();
}

} // namespace server
