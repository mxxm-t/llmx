#pragma once
// The routes of docs/SERVER.md: native /v1/generate, /v1/chat, /v1/tokenize, /v1/detokenize and /v1/health, and the OpenAI-compatible /v1/chat/completions, /v1/completions and /v1/models.
// Both families share one parse, one scheduler request and one drain loop; one thread per connection parses, tokenizes, submits and drains.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include "core/json.hpp"
#include "config.hpp"
#include "core/utf8.hpp"
#include "inference/chat.hpp"
#include "model/place.hpp"
#include "server/context_cut.hpp"
#include "server/http.hpp"
#include "server/scheduler.hpp"

namespace server {

// The host memory for donors the devices evict when the server is given none (--host-cache-bytes): what `max_seqs` conversations take at the most one request may hold, within half of what is free.
// Each copy still leaves the host the pool's fixed reserve (Model::host_admits).
inline size_t default_host_cache(const infer::Model& model, size_t max_seqs) {
    if (!model.caches_on_devices()) return 0;
    const size_t bt = model.kv_block_tokens(), limit = std::min((size_t)model.context_length(), model.kv_tokens_total()) / bt * bt;
    return host_cache_default(model.host_bytes(limit), max_seqs, core::host_memory_available());
}

// The start line for uncapped requests that fill the pool and will not take turns here, or not all of them, though turns were possible or asked for; empty where they will, and on the CPU with no host memory given.
// The host memory is asked first: given enough, a server takes turns wherever its caches are (docs/SERVER.md, turns at the pool's edge).
inline std::string no_turns(const infer::Model& model, size_t host_cap, bool given) {
    const size_t bt = model.kv_block_tokens(), need = model.host_bytes(std::min((size_t)model.context_length(), model.kv_tokens_total()) / bt * bt);
    if (!given && !model.caches_on_devices()) return "";
    const size_t tokens = host_cap ? turn_tokens(host_cap, bt, std::min((size_t)model.context_length(), model.kv_tokens_total()), [&](size_t n) { return model.host_bytes(n); }) : 0;
    return turns_line(host_cap, need, tokens, model.keeps_state(), model.checkpoint_slots());
}

struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    size_t max_seqs = 16;
    size_t max_queue = 64;   // requests waiting for admission; past it, 503
    size_t passes = 0;       // passes in flight; 0 takes the stage count on a pipelined layer split and one elsewhere (Scheduler)
    double generating_share = 0;   // of each device's time, a fixed share generating requests keep while prompts are read (Scheduler); 0 leaves it to the automatic rule
    bool no_pacing = false;        // generating requests are served with the prompts' passes alone
    Pacing pacing;                 // what the two flags and the placement give (resolve_pacing)
    std::string unpaced;           // and where that is off, why
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
};

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

class Api {
public:
    Api(infer::Model& model, const bpe::Tokenizer& tok, const chat::ChatFormat& format, Scheduler& sched,
        const Config& cfg)
        : model_(model), tok_(tok), format_(format), sched_(sched), cfg_(cfg), started_((int64_t)std::time(nullptr)) {
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
        std::string devices, positions, sources;
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
                  ",\"context_tokens\":" + n(model_.context_length()) + ",\"kv_tokens\":" + n(model_.kv_tokens_total()) + ",\"devices\":[" + devices + "]}" +
                  ",\"precision\":" + dtype_json(cfg_.dtype) +
                  ",\"requests\":{\"now\":{\"active\":" + n(s.active) + ",\"queued\":" + n(s.queued) + ",\"paused\":" + n(s.paused) + ",\"kv_tokens\":" + n(s.kv_tokens) +
                  "},\"limits\":{\"active\":" + n(cfg_.max_seqs) + ",\"queued\":" + n(cfg_.max_queue) +
                  "},\"since_start\":{\"finished\":" + n(s.finished) + ",\"prompt_tokens\":" + n(s.prompt_tokens) + ",\"generated_tokens\":" + n(s.generated_tokens) +
                  ",\"shifts\":" + n(shifts_.load()) + ",\"shifted_tokens\":" + n(shifted_tokens_.load()) + "}}" +
                  ",\"reuse\":{\"since_start\":{\"forks\":" + n(s.prefix_hits) + ",\"tokens\":" + n(s.prefix_tokens) +
                  "},\"device\":{\"now\":{\"entries\":" + n(s.donors) + ",\"state_checkpoints\":" + n(s.checkpoints) + ",\"kv_tokens\":" + n(s.kv_donor_tokens) + "}}" +
                  ",\"host\":{\"now\":{\"entries\":" + n(s.host_donors) + ",\"bytes\":" + n(s.host_bytes) + ",\"limit_bytes\":" + n(s.host_limit) + ",\"allocated_bytes\":" + n(s.host_allocated) +
                  "},\"since_start\":{\"promotions\":" + n(s.host_hits) + ",\"yielded_bytes\":" + n(s.host_yielded) + ",\"bytes_moved\":" + n(s.host_bytes_moved) + "}}" +
                  ",\"disk\":{\"now\":{\"entries\":" + n(s.disk_entries) + ",\"segments\":" + n(s.disk_entries - s.disk_states) + ",\"states\":" + n(s.disk_states) + ",\"bytes\":" + n(s.disk_bytes) + ",\"limit_bytes\":" + n(s.disk_limit) +
                  ",\"unproved\":" + n(s.disk_unproved) + ",\"in_flight\":" + n(s.disk_in_flight) + ",\"ready\":" + yes(s.disk_ready) + ",\"writing\":" + yes(s.disk_writing) +
                  "},\"since_start\":{\"hits\":" + n(s.disk_hits) + ",\"bytes_read\":" + n(s.disk_bytes_read) + ",\"bytes_written\":" + n(s.disk_bytes_written) +
                  ",\"waits\":" + n(s.disk_waits) + ",\"wait_ms\":" + n((uint64_t)s.disk_wait_ms) + ",\"errors\":" + n(s.disk_errors) +
                  ",\"refused\":" + n(s.disk_refused) + ",\"dropped_for_cap\":" + n(s.disk_capped) + ",\"proved\":" + n(s.disk_proved) + ",\"disproved\":" + n(s.disk_disproved) + ",\"lost_before_written\":" + n(s.lost_before_written) + "}}" +
                  ",\"boundaries\":{\"now\":{\"entries\":" + n(s.boundaries) + "},\"since_start\":{\"hits\":" + n(s.boundary_hits) + "}},\"moves\":{\"since_start\":{" + moves(s) + "}}}" +
                  ",\"pressure\":{\"since_start\":{\"pauses\":" + n(s.pauses) + ",\"stalls\":" + n(s.stalls) + ",\"waits\":" + n(s.waits) +
                  ",\"recomputed_tokens\":" + n(s.recomputed) + ",\"resumes_taking_history_back\":" + n(s.taken_back) + ",\"turns\":" + n(s.yields) + "}}" +
                  ",\"reread\":{\"since_start\":{\"jobs\":" + n(s.reprefills) + ",\"rows\":" + n(s.reprefill_rows) + ",\"cancelled\":" + n(s.reprefill_cancels) + ",\"handed\":" + n(s.reprefill_handed) + ",\"rows_released\":" + n(s.reprefill_released) + "}}" +
                  ",\"drafting\":{\"sources\":[" + sources + "],\"since_start\":{\"drafted\":" + n(drafted) + ",\"kept\":" + n(kept) + ",\"failed\":" + n(s.draft_failures) + ",\"by_position\":[" + positions + "]" +
                  ",\"own_history\":{\"drafted\":" + n(own_drafted) + ",\"kept\":" + n(own_kept) + "}}}" +
                  ",\"passes\":{\"limit\":" + n(s.passes) + ",\"in_flight\":" + n(s.in_flight) + ",\"sampling_threads\":" + n(s.samplers) + ",\"since_start\":{\"apart\":" + n(s.apart) + ",\"between_pieces\":" + n(s.between) + ",\"rode\":" + n(s.rode) + "}}" +
                  ",\"pacing\":{\"mode\":\"" + (s.automatic ? "automatic" : s.share > 0 ? "fixed" : "off") + "\",\"target_gap_ms\":" + n((uint64_t)kServiceGapMs) +
                  ",\"since_start\":{\"gaps\":" + n(s.gaps) + ",\"gaps_above_target\":" + n(s.gaps_late) + ",\"deferred_by_floor\":" + n(s.floor_deferred) +
                  ",\"longest_piece_ms\":" + n((uint64_t)s.longest_piece_ms) + "}}" +
                  (s.timed ? ",\"timing\":" + timing_json(s.timing) : std::string()) + "}");
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

    // A message's content: a string, or the array of text parts the compatible chat route accepts.
    // An assistant message that carries tool_calls may leave its content null or out, which renders as an empty content.
    static std::string content_of(const jmini::Value& m) {
        const jmini::Value* content = m.get("content");
        if ((!content || content->t == jmini::Value::T::Null) && given(m, "tool_calls")) return {};
        if (!content) throw BadRequest(400, "every message needs a content");
        if (content->isString()) return content->asString();
        if (!content->isArray()) throw BadRequest(400, "message content must be a string or an array of text parts");
        std::string text;
        for (const auto& part : content->asArray()) {
            const jmini::Value* type = part.get("type");
            const jmini::Value* t = part.get("text");
            if (!type || !type->isString() || type->asString() != "text" || !t || !t->isString())
                throw BadRequest(400, "only text content parts are supported");
            text += t->asString();
        }
        return text;
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
            if (reasoning && reasoning->isString()) messages.push_back({role, content_of(m), reasoning->asString()});
            else if (role == "assistant") messages.push_back(format_.assistant(content_of(m)));
            else messages.push_back({role, content_of(m), std::nullopt});
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
            // A parameter whose declared type is not a string, nor a list holding "string", has its value written as JSON (chat::ToolCallSplit).
            const jmini::Value* properties = params ? params->get("properties") : nullptr;
            if (properties && properties->isObject())
                for (const auto& [key, schema] : properties->obj) {
                    const jmini::Value* kind = schema.get("type");
                    bool text = !kind || (kind->isString() && kind->asString() == "string");
                    if (kind && kind->isArray())
                        for (const auto& k : kind->asArray()) text = text || (k.isString() && k.asString() == "string");
                    if (!text) t.typed[name->asString()].insert(key);
                }
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
    size_t stable_of(const jmini::Value& body, Route route, const std::vector<uint32_t>& prompt) const {
        if (!model_.checkpoint_slots() || (route != Route::chat && route != Route::chat_completions)) return prompt.size();
        return chat::stable_prefix(format_, tok_, messages_of(body), prompt, template_vars(body));
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
        size_t n = 0, at = 0;
        for (uint32_t id : ids) {
            at += tok_.decode({id}).size();
            if (at > shared) break;
            ++n;
        }
        return n;
    }

    // The ids a chat request's next turn begins with after `text`, its reply as far as it is written, given back as a client of the route sends it (reply_sent_back, docs/SPECULATIVE.md, section 2, Idle re-prefill).
    // None while the reply's reasoning or a call is still being written, which a next turn may drop or would write otherwise.
    std::vector<uint32_t> next_turn(const jmini::Value& body, Route route, const std::string& prompt, const std::string& text, bool writing) const {
        const bool opened = chat::opens_reasoning(prompt);
        if (writing && (opened || text.find("<think>") != std::string::npos) && text.find("</think>") == std::string::npos) return {};
        std::vector<chat::Message> messages = messages_of(body);
        const Tools tools = tools_of(body);
        bool open = false;
        messages.push_back(reply_sent_back(format_, route == Route::chat_completions, opened, text,
                                           route == Route::chat_completions && tools.on ? format_.tool_format : chat::ToolFormat::none, tools.typed, tools.single, &open));
        if (writing && open) return {};
        return chat::stable_prefix(format_, tok_, messages, writing, template_vars(body));
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
    static std::string timings_json(const Request& r, size_t tokens) {
        const Request::Timings t = r.timings();
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
                           const std::string& logprobs) const {
        std::string c = chunk(route, id, "", false, &reason, logprobs);
        c.pop_back();
        return c + ",\"timings\":" + timings_json(r, tokens) + "}";
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
        const std::vector<uint32_t> ids = encode(chat ? render_messages(body) : text->asString());
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

    // What a cut dropped of a prompt, for the reply's context object and the request's line.
    struct Fitted {
        size_t messages = 0, tokens = 0, renders = 0;   // renders: those that placed the messages' ends, on a chat route
        bool cut = false, at_marker = false;
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
            if (bytes->empty()) {
                size_t at = 0;
                for (uint32_t id : ids) bytes->push_back(at += tok_.decode({id}).size());
            }
            // A part ends with its last text, as its end token ends it, not with the whitespace a template writes after the token.
            size_t len = text.size();
            while (len > 0 && (text[len - 1] == '\n' || text[len - 1] == ' ')) --len;
            return (size_t)(std::upper_bound(bytes->begin(), bytes->end(), len) - bytes->begin());
        });
    }

    // A prompt that does not fit what a request may hold (docs/SERVER.md, A prompt larger than the context): refused with its numbers, or, with --context-overflow shift, cut at a cut point in steps of half of that (context_cut.hpp).
    // A chat route drops its oldest messages after the system message and a text route rows after its leading block (cut_rows), and `body`, `prompt` and `ids` are the cut request's afterwards.
    Fitted fit(jmini::Value& body, Route route, const SampleParams& params, std::string& prompt, std::vector<uint32_t>& ids) {
        // The rows the prompt may take beside its reply, the scheduler's rule, and the step a cut drops in.
        const size_t limit = sched_.token_limit(), n = ids.size(), room = sched_.prompt_room(params), step = limit / kCutTo;
        const auto fits = [&](size_t rows) { return rows <= room; };
        if (fits(n)) return {};
        const auto refuse = [&](const std::string& why) { throw BadRequest(413, sched_.too_long(n, params) + "; " + why); };
        if (!cfg_.context_shift)
            refuse(std::string(params.until_limit ? "shorten it, send max_tokens" : "shorten it") + ", or start the server with --context-overflow shift, which drops the oldest turns");
        Fitted f;
        f.cut = true;
        if (route == Route::chat || route == Route::chat_completions) {
            const std::vector<chat::Message> messages = messages_of(body);
            const std::vector<chat::TemplateVar> vars = template_vars(body);
            MessageEnds ends = message_ends(messages, vars, ids);
            std::vector<char> starts;
            for (const chat::Message& m : messages) starts.push_back(m.role == "user");
            const size_t lead = messages[0].role == "system" ? 1 : 0, k = first_kept(ends, starts, lead, n, room, step);
            f.renders = ends.renders() + 1;
            if (k == messages.size()) refuse("its newest turns alone do not fit, so no cut helps");
            for (auto& member : body.obj)
                if (member.first == "messages") member.second.arr.erase(member.second.arr.begin() + (std::ptrdiff_t)lead, member.second.arr.begin() + (std::ptrdiff_t)k);
            prompt = prompt_of(body, route);
            ids = encode(prompt);
            f.messages = k - lead;
            f.at_marker = true;
        } else {
            const RowCut c = cut_rows(cut_points(ids.data(), n, [this](uint32_t id) { return tok_.is_eos(id); }), n, room, step);
            if (c.from >= n) refuse("its newest rows alone do not fit, so no cut helps");
            ids.erase(ids.begin() + (std::ptrdiff_t)c.lead, ids.begin() + (std::ptrdiff_t)c.from);
            f.at_marker = c.at_marker;
        }
        f.tokens = n - ids.size();
        if (!fits(ids.size())) refuse("cut to " + std::to_string(ids.size()) + " it still does not fit with what it asks for");
        ++shifts_;
        shifted_tokens_ += f.tokens;
        const std::string messages = f.messages ? ", " + std::to_string(f.messages) + " messages" : std::string();
        const std::string renders = f.renders ? ", its messages' ends found in " + std::to_string(f.renders) + " render" + (f.renders == 1 ? "" : "s") : std::string();
        std::fprintf(stderr, "server: a prompt of %zu tokens cut to %zu to fit %zu, %zu tokens dropped%s%s%s\n", n, ids.size(), limit, f.tokens,
                     messages.c_str(), renders.c_str(), f.at_marker ? "" : ", not at a turn's end");
        return f;
    }

    // A reply's context object: the rows the request holds against what it may, and what a cut dropped of its prompt.
    std::string context_json(size_t prompt_tokens, size_t tokens, const Fitted& f) const {
        std::string c = "{\"tokens\":" + std::to_string(prompt_tokens + tokens) + ",\"limit\":" + std::to_string(sched_.token_limit());
        if (f.cut) {
            if (f.messages) c += ",\"dropped_messages\":" + std::to_string(f.messages);
            c += ",\"dropped_tokens\":" + std::to_string(f.tokens) + ",\"at_marker\":" + (f.at_marker ? "true" : "false");
        }
        return c + "}";
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
        std::vector<uint32_t> ids = encode(prompt);
        const Fitted fitted = fit(body, route, params, prompt, ids);
        const size_t stable = stable_of(body, route, ids), start = message_start_of(body, route, prompt, ids);
        try { r = sched_.submit(std::move(ids), params, stable, start, received); }
        catch (const TooLong& e) { throw BadRequest(413, e.what()); }
        catch (const QueueFull& e) { throw BadRequest(503, e.what()); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }
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
            if (following) sched_.follow(r, ended ? next_turn(body, route, prompt, text, false) : std::vector<uint32_t>{}, true);
            following = false;
        };
        try {
            if (stream) c.begin_stream(200, "text/event-stream");
            Request::Token tok;
            Request::Clock::time_point probe = Request::Clock::now() + kProbe;
            for (;;) {
                const Request::Next got = r->next(tok, probe);
                if (got == Request::Next::end) break;
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
                if (following && gen.size() % kFollowEvery == 0) sched_.follow(r, next_turn(body, route, prompt, text, true), false);
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
            const size_t prompt_tokens = r->prompt_tokens();
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
                c.write_chunk("data: " + last_chunk(route, id, finish_reason(finish, caller.calls() > 0), *r, gen.size(), no_logprobs) + "\n\n");
                if (include_usage)
                    c.write_chunk("data: " + head(id, route == Route::chat_completions ? "chat.completion.chunk" : "text_completion") +
                                  ",\"choices\":[],\"usage\":" + usage_json(prompt_tokens, gen.size(), r->reused()) + ",\"context\":" + context_json(prompt_tokens, gen.size(), fitted) + "}\n\n");
                c.write_chunk("data: [DONE]\n\n");
                c.end_stream();
            } else if (stream) {
                if (!pending.empty()) c.write_chunk("data: {\"text\":" + jmini::quote(pending) + "}\n\n");
                c.write_chunk("data: {\"done\":true,\"finish\":" + jmini::quote(finish) +
                              ",\"tokens\":" + std::to_string(gen.size()) + ",\"context\":" + context_json(prompt_tokens, gen.size(), fitted) + "}\n\ndata: [DONE]\n\n");
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
                          usage_json(prompt_tokens, gen.size(), r->reused()) + ",\"context\":" + context_json(prompt_tokens, gen.size(), fitted) + ",\"timings\":" + timings_json(*r, gen.size()) + "}");
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
                          std::to_string(prompt_tokens) + ",\"reused_tokens\":" + std::to_string(r->reused()) +
                          ",\"tokens\":" + std::to_string(gen.size()) + ",\"context\":" + context_json(prompt_tokens, gen.size(), fitted) + "}");
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

// How generating requests are served beside prompts, from the two flags and why the placement feeds no stage in pieces (infer::unpaced), into cfg.pacing and cfg.unpaced.
// The automatic rule is taken wherever a stage is fed in pieces and is off, with the reason, elsewhere; a fixed share is the flag's own demand, refused by the flag's name where it cannot be kept.
inline void resolve_pacing(Config& cfg, const std::string& unpaced) {
    if (cfg.generating_share > 0 && !unpaced.empty()) throw std::runtime_error("--generating-share: " + unpaced);
    cfg.unpaced = cfg.no_pacing ? "--no-pacing" : unpaced;
    cfg.pacing = cfg.no_pacing ? Pacing{} : cfg.generating_share > 0 ? Pacing::fixed(cfg.generating_share) : unpaced.empty() ? Pacing::automatic() : Pacing{};
}
// The start line that says which.
inline std::string pacing_line(const Config& cfg) {
    if (cfg.pacing.mode == Pacing::Mode::fixed) return "server: pacing: fixed share " + jmini::number((float)cfg.pacing.share) + "\n";
    if (cfg.pacing.mode == Pacing::Mode::off) return "server: pacing: off (" + cfg.unpaced + ")\n";
    return "server: pacing: automatic (while a prompt is read a generating request is served about every " + std::to_string((int)kServiceGapMs) +
           " ms, where the device's steps allow)\n";
}

// Serve until the listener is closed, or a signal stops it (detail::StopOnSignal): the scheduler on its own thread, the accept loop here, one detached thread per connection.
inline void serve(infer::Model& model, const bpe::Tokenizer& tok, const chat::ChatFormat& format,
                  const Config& cfg, http::Listener& listener) {
    Scheduler sched(model, tok, cfg.max_seqs, cfg.max_queue, cfg.passes, cfg.timing, cfg.host_cache_bytes.value_or(default_host_cache(model, cfg.max_seqs)), cfg.proposer,
                    cfg.draft_max, true, cfg.disk, cfg.pacing);
    const Scheduler::Stats started = sched.stats();
    std::fprintf(stderr, "server: up to %zu pass%s in flight over %zu stage%s, %zu sampling thread%s beside the scheduler's\n", started.passes,
                 started.passes == 1 ? "" : "es", model.stage_count(), model.stage_count() == 1 ? "" : "s", started.samplers,
                 started.samplers == 1 ? "" : "s");
    std::fputs(pacing_line(cfg).c_str(), stderr);
    Api api(model, tok, format, sched, cfg);
    std::thread runner([&] { sched.run(); });
    const detail::StopOnSignal stop_on_signal(listener);
    std::atomic<int> open{0};
    for (;;) {
        http::Connection c = listener.accept();
        if (!c.open()) break;
        ++open;
        std::thread([&api, &open](http::Connection conn) {
            try { api.handle(conn); } catch (...) {}
            --open;
        }, std::move(c)).detach();
    }
    sched.stop();
    runner.join();
    // Connection threads still draining end with their requests cancelled.
    while (open.load() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

} // namespace server
