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
#include "core/utf8.hpp"
#include "inference/chat.hpp"
#include "model/place.hpp"
#include "server/http.hpp"
#include "server/scheduler.hpp"

namespace server {

struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    size_t max_seqs = 16;
    size_t max_queue = 64;   // requests waiting for admission; past it, 503
    size_t passes = 0;       // passes in flight; 0 takes the stage count on a pipelined layer split and one elsewhere (Scheduler)
    int state_checkpoints = -1;   // on a model that keeps a state, the states kept for prefix reuse; -1 for the most the fit gives up to max_seqs
    bool timing = false;     // time the rounds and the stages for /v1/health, over backends made to time their work
    std::optional<size_t> host_cache_bytes;   // host memory for donors the devices evict (Scheduler); none given takes default_host_cache
    std::string model_name;
    infer::DtypePlan dtype;
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

    void health(http::Connection& c) {
        const Scheduler::Stats s = sched_.stats();
        c.respond(200, "application/json",
                  "{\"status\":\"ok\",\"model\":" + jmini::quote(cfg_.model_name) + ",\"dtype\":" + dtype_json(cfg_.dtype) +
                  ",\"active\":" + std::to_string(s.active) + ",\"queued\":" + std::to_string(s.queued) +
                  ",\"donors\":" + std::to_string(s.donors) + ",\"prefix_hits\":" + std::to_string(s.prefix_hits) +
                  ",\"prefix_tokens\":" + std::to_string(s.prefix_tokens) + ",\"pauses\":" + std::to_string(s.pauses) +
                  ",\"paused\":" + std::to_string(s.paused) + ",\"stalls\":" + std::to_string(s.stalls) + ",\"waits\":" + std::to_string(s.waits) +
                  ",\"recomputed\":" + std::to_string(s.recomputed) + ",\"taken_back\":" + std::to_string(s.taken_back) +
                  ",\"checkpoints\":" + std::to_string(s.checkpoints) + ",\"host_donors\":" + std::to_string(s.host_donors) +
                  ",\"host_bytes\":" + std::to_string(s.host_bytes) + ",\"host_hits\":" + std::to_string(s.host_hits) +
                  ",\"host_bytes_moved\":" + std::to_string(s.host_bytes_moved) + ",\"reprefills\":" + std::to_string(s.reprefills) +
                  ",\"reprefill_rows\":" + std::to_string(s.reprefill_rows) + ",\"reprefill_cancels\":" + std::to_string(s.reprefill_cancels) +
                  ",\"passes\":" + std::to_string(s.passes) + ",\"in_flight\":" + std::to_string(s.in_flight) +
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
    // A timed scheduler's figures (--timing): each of the thread's times as a mean over the rounds, each stage's idle share over the span its device time was read in, and the device-bound rate, the rows the passes in that span carried over the busiest stage's device time.
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
               ",\"receive_wait_ms\":" + mean(t.receive_wait_ms) + ",\"staging_wait_ms\":" + mean(t.staging_wait_ms) +
               ",\"open_wait_ms\":" + mean(t.open_wait_ms) + ",\"logits_wait_ms\":" + mean(t.logits_wait_ms) + ",\"stage_idle\":[" + idle +
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
    static std::string content_of(const jmini::Value& m) {
        const jmini::Value* content = m.get("content");
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

    // A request's messages as chat records its own turns: a message's reasoning as the client passes it in `reasoning_content`, or else an assistant message as chat::ChatFormat::assistant keeps chat's own replies, so a conversation renders as chat renders it.
    std::vector<chat::Message> messages_of(const jmini::Value& body) const {
        const jmini::Value* msgs = body.get("messages");
        if (!msgs || !msgs->isArray() || msgs->asArray().empty()) throw BadRequest(400, "messages must be a non-empty array");
        std::vector<chat::Message> messages;
        for (const auto& m : msgs->asArray()) {
            const jmini::Value* role = m.get("role");
            if (!role || !role->isString()) throw BadRequest(400, "every message needs a string role");
            const jmini::Value* reasoning = m.get("reasoning_content");
            if (reasoning && reasoning->t != jmini::Value::T::Null && !reasoning->isString()) throw BadRequest(400, "reasoning_content must be a string");
            if (reasoning && reasoning->isString()) messages.push_back({role->asString(), content_of(m), reasoning->asString()});
            else if (role->asString() == "assistant") messages.push_back(format_.assistant(content_of(m)));
            else messages.push_back({role->asString(), content_of(m), std::nullopt});
        }
        return messages;
    }

    // A request's chat_template_kwargs: variables of a boolean, number, string or null the template reads beside the conversation, none of them one the render itself sets.
    static std::vector<chat::TemplateVar> template_vars(const jmini::Value& body) {
        std::vector<chat::TemplateVar> vars;
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
            else if (v.isNumber())
                vars.push_back({name, v.num == std::floor(v.num) && std::fabs(v.num) < 9007199254740992.0 ? chat::jj::Value::integer((int64_t)v.num) : chat::jj::Value::number(v.num)});
            else throw BadRequest(400, "chat_template_kwargs values must be booleans, numbers, strings or null");
        }
        return vars;
    }

    // How much of a request's prompt a follow-up turn begins with, where the model keeps a state a follow-up could fork (chat::stable_prefix for a chat route, a text's whole prompt); a model that keeps none is not asked to render again.
    size_t stable_of(const jmini::Value& body, Route route, const std::vector<uint32_t>& prompt) const {
        if (!model_.checkpoint_slots() || (route != Route::chat && route != Route::chat_completions)) return prompt.size();
        return chat::stable_prefix(format_, tok_, messages_of(body), prompt, template_vars(body));
    }

    // The ids a chat request's next turn begins with after `text`, its reply as far as it is written, given back as the route gave it to the client: the content apart from the reasoning on the compatible route, the text whole on the native one (docs/SPECULATIVE.md, section 2, Idle re-prefill).
    // None while the reply's reasoning is still being written, which a next turn may drop.
    std::vector<uint32_t> next_turn(const jmini::Value& body, Route route, const std::string& prompt, const std::string& text, bool writing) const {
        const bool opened = chat::opens_reasoning(prompt);
        if (writing && (opened || text.find("<think>") != std::string::npos) && text.find("</think>") == std::string::npos) return {};
        std::vector<chat::Message> messages = messages_of(body);
        if (route == Route::chat_completions) {
            chat::ReplySplit split(opened);
            const chat::ReplySplit::Parts parts = split.feed(text), last = split.finish();
            std::optional<std::string> reasoning;
            if (split.reasoned()) reasoning = parts.reasoning + last.reasoning;
            messages.push_back({"assistant", parts.content + last.content, reasoning});
        } else {
            messages.push_back(format_.assistant(text));
        }
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
        if (!p || !p->isString() || p->asString().empty()) throw BadRequest(400, "prompt must be a non-empty string");
        return p->asString();
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
    // The logprobs of sampled tokens in a compatible route's shape: the chat route's content list beside a null refusal, or the completions route's parallel lists, whose top_logprobs map each listed token's text to its log-probability and hold the sampled token too.
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

    static std::string finish_reason(const std::string& finish) {
        return finish == "length" ? "length" : "stop";
    }
    std::string usage_json(size_t prompt_tokens, size_t tokens) const {
        return "{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(tokens) +
               ",\"total_tokens\":" + std::to_string(prompt_tokens + tokens) + "}";
    }
    // A finished request's speed in the fields clients that display speed read beside the standard usage: prompt tokens prefilled and reused, milliseconds to the first token, and generation after it.
    static std::string timings_json(const Request& r, size_t tokens) {
        const Request::Timings t = r.timings();
        const size_t cached = r.reused(), prefilled = r.prompt_tokens() - cached;
        auto num = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.3f", v); return std::string(b); };
        return "{\"cache_n\":" + std::to_string(cached) + ",\"prompt_n\":" + std::to_string(prefilled) +
               ",\"prompt_ms\":" + num(t.prompt_ms) + ",\"prompt_per_second\":" + num(Request::Timings::per_second(prefilled, t.prompt_ms)) +
               ",\"predicted_n\":" + std::to_string(Request::Timings::predicted(tokens)) + ",\"predicted_ms\":" + num(t.predicted_ms) +
               ",\"predicted_per_second\":" + num(t.predicted_per_second(tokens)) + ",\"queued_ms\":" + num(t.queued_ms) + "}";
    }
    // The head every compatible object and chunk starts with.
    std::string head(const std::string& id, const char* object) const {
        return "{\"id\":" + jmini::quote(id) + ",\"object\":\"" + object + "\",\"created\":" + std::to_string(started_) +
               ",\"model\":" + jmini::quote(cfg_.model_name);
    }
    // One streamed chunk of a compatible route: a chat delta or a text piece, with the finish reason on the last, and the chunk's logprobs when the request asked for them.
    // A chat delta carries `reasoning` as reasoning_content beside its content when a reply's reasoning is split off.
    std::string chunk(Route route, const std::string& id, const std::string& piece, bool first,
                      const std::string* finish, const std::string& logprobs = "", const std::string* reasoning = nullptr) const {
        const std::string fr = finish ? jmini::quote(finish_reason(*finish)) : "null";
        const std::string lp = logprobs.empty() ? "" : ",\"logprobs\":" + logprobs;
        if (route == Route::chat_completions) {
            const std::string thought = reasoning ? "\"reasoning_content\":" + jmini::quote(*reasoning) + "," : "";
            std::string delta = first ? "{\"role\":\"assistant\"," + thought + "\"content\":" + jmini::quote(piece) + "}"
                              : finish ? "{}" : "{" + thought + "\"content\":" + jmini::quote(piece) + "}";
            return head(id, "chat.completion.chunk") + ",\"choices\":[{\"index\":0,\"delta\":" + delta + lp +
                   ",\"finish_reason\":" + fr + "}]}";
        }
        return head(id, "text_completion") + ",\"choices\":[{\"index\":0,\"text\":" + jmini::quote(piece) + lp +
               ",\"finish_reason\":" + fr + "}]}";
    }
    // The same chunk with a request's timings, for the one that carries the finish reason.
    std::string last_chunk(Route route, const std::string& id, const std::string& finish, const Request& r, size_t tokens,
                           const std::string& logprobs) const {
        std::string c = chunk(route, id, "", false, &finish, logprobs);
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

    void generate(http::Connection& c, const http::Request& req, Route route) {
        const jmini::Value body = body_of(req);
        const std::string prompt = prompt_of(body, route);
        const SampleParams params = params_of(body, route);
        const bool stream = flag(body, "stream");
        const jmini::Value* so = body.get("stream_options");
        const bool include_usage = so && so->isObject() && flag(*so, "include_usage");

        std::shared_ptr<Request> r;
        std::vector<uint32_t> ids = encode(prompt);
        const size_t stable = stable_of(body, route, ids);
        try { r = sched_.submit(std::move(ids), params, stable); }
        catch (const TooLong& e) { throw BadRequest(413, e.what()); }
        catch (const QueueFull& e) { throw BadRequest(503, e.what()); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }
        const std::string id = (route == Route::chat_completions ? "chatcmpl-" : "cmpl-") + std::to_string(next_id_.fetch_add(1));
        // A chat reply gives its reasoning as reasoning_content, apart from its content, as clients show a reasoning model's: inside the <think> its template opened or it opened itself.
        const bool split = route == Route::chat_completions;
        chat::ReplySplit splitter(chat::opens_reasoning(prompt));

        // Drain the channel.
        // A write that fails means the client went away: cancel the request and stop.
        // Between writes the socket is looked at every kProbe, token or not, since a request that is queued, prefilling or building a whole reply writes nothing that could fail.
        std::vector<uint32_t> gen;
        std::string text, pending;
        bool first = true;
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
                if (got == Request::Next::timeout) continue;
                gen.push_back(tok.id);
                std::string bytes = tok_.decode({tok.id});
                pending += bytes;
                const size_t whole = utf8_complete(pending);
                const std::string piece = utf8_sanitize(pending.substr(0, whole));
                pending.erase(0, whole);
                text += piece;
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
                    if (split) {
                        const chat::ReplySplit::Parts parts = splitter.feed(piece);
                        c.write_chunk("data: " + chunk(route, id, parts.content, first, nullptr, lp, parts.reasoning.empty() ? nullptr : &parts.reasoning) + "\n\n");
                    } else {
                        c.write_chunk("data: " + chunk(route, id, piece, first, nullptr, lp) + "\n\n");
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
            text += pending;
            const std::string finish = r->finish();
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
                    chat::ReplySplit::Parts parts = splitter.feed(pending);
                    const chat::ReplySplit::Parts last = splitter.finish();
                    parts.reasoning += last.reasoning;
                    parts.content += last.content;
                    if (!parts.reasoning.empty() || !parts.content.empty() || first)
                        c.write_chunk("data: " + chunk(route, id, parts.content, first, nullptr, no_logprobs, parts.reasoning.empty() ? nullptr : &parts.reasoning) + "\n\n");
                } else if (!pending.empty() || first) {
                    c.write_chunk("data: " + chunk(route, id, pending, first, nullptr, no_logprobs) + "\n\n");
                }
                c.write_chunk("data: " + last_chunk(route, id, finish, *r, gen.size(), no_logprobs) + "\n\n");
                if (include_usage)
                    c.write_chunk("data: " + head(id, route == Route::chat_completions ? "chat.completion.chunk" : "text_completion") +
                                  ",\"choices\":[],\"usage\":" + usage_json(prompt_tokens, gen.size()) + "}\n\n");
                c.write_chunk("data: [DONE]\n\n");
                c.end_stream();
            } else if (stream) {
                if (!pending.empty()) c.write_chunk("data: {\"text\":" + jmini::quote(pending) + "}\n\n");
                c.write_chunk("data: {\"done\":true,\"finish\":" + jmini::quote(finish) +
                              ",\"tokens\":" + std::to_string(gen.size()) + "}\n\ndata: [DONE]\n\n");
                c.end_stream();
            } else if (compat(route)) {
                const std::string lp = params.logprobs ? ",\"logprobs\":" + compat_logprobs(route, sampled.data(), sampled.size()) : "";
                std::string message = "\"content\":" + jmini::quote(text);
                if (split) {
                    const chat::ReplySplit::Parts parts = splitter.feed(text);
                    const chat::ReplySplit::Parts last = splitter.finish();
                    message = (splitter.reasoned() ? "\"reasoning_content\":" + jmini::quote(parts.reasoning + last.reasoning) + "," : std::string()) +
                              "\"content\":" + jmini::quote(parts.content + last.content);
                }
                const std::string choice = route == Route::chat_completions
                    ? "{\"index\":0,\"message\":{\"role\":\"assistant\"," + message + "}"
                    : "{\"index\":0,\"text\":" + jmini::quote(text);
                c.respond(200, "application/json",
                          head(id, route == Route::chat_completions ? "chat.completion" : "text_completion") + ",\"choices\":[" + choice + lp +
                          ",\"finish_reason\":" + jmini::quote(finish_reason(finish)) + "}],\"usage\":" +
                          usage_json(prompt_tokens, gen.size()) + ",\"timings\":" + timings_json(*r, gen.size()) + "}");
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
                          ",\"tokens\":" + std::to_string(gen.size()) + "}");
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
    std::atomic<uint64_t> next_id_{1};
};

// Serve until the listener is closed: the scheduler on its own thread, the accept loop here, one detached thread per connection.
inline void serve(infer::Model& model, const bpe::Tokenizer& tok, const chat::ChatFormat& format,
                  const Config& cfg, http::Listener& listener) {
    Scheduler sched(model, tok, cfg.max_seqs, cfg.max_queue, cfg.passes, cfg.timing, cfg.host_cache_bytes.value_or(default_host_cache(model, cfg.max_seqs)));
    const Scheduler::Stats started = sched.stats();
    std::fprintf(stderr, "server: up to %zu pass%s in flight over %zu stage%s, %zu sampling thread%s beside the scheduler's\n", started.passes,
                 started.passes == 1 ? "" : "es", model.stage_count(), model.stage_count() == 1 ? "" : "s", started.samplers,
                 started.samplers == 1 ? "" : "s");
    Api api(model, tok, format, sched, cfg);
    std::thread runner([&] { sched.run(); });
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
