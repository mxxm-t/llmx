#pragma once
// The routes of docs/SERVER.md: native /v1/generate, /v1/chat, /v1/tokenize, /v1/detokenize and /v1/health, and the OpenAI-compatible /v1/chat/completions, /v1/completions and /v1/models.
// Both families share one parse, one scheduler request and one drain loop; one thread per connection parses, tokenizes, submits and drains.
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
#include "server/http.hpp"
#include "server/scheduler.hpp"

namespace server {

struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    size_t max_seqs = 16;
    size_t max_queue = 64;   // requests waiting for admission; past it, 503
    std::string model_name;
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
                  "{\"status\":\"ok\",\"model\":" + jmini::quote(cfg_.model_name) +
                  ",\"active\":" + std::to_string(s.active) + ",\"queued\":" + std::to_string(s.queued) +
                  ",\"donors\":" + std::to_string(s.donors) + ",\"prefix_hits\":" + std::to_string(s.prefix_hits) +
                  ",\"prefix_tokens\":" + std::to_string(s.prefix_tokens) + ",\"pauses\":" + std::to_string(s.pauses) + "}");
    }
    // The list clients read the model id from, with the file's context length and vocabulary beside the standard fields.
    void models(http::Connection& c) {
        c.respond(200, "application/json",
                  "{\"object\":\"list\",\"data\":[{\"id\":" + jmini::quote(cfg_.model_name) +
                  ",\"object\":\"model\",\"created\":" + std::to_string(started_) + ",\"owned_by\":\"llmx\"" +
                  ",\"context_length\":" + std::to_string(model_.config().context_length) +
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

    // A render the template itself fails, such as its raise_exception on a conversation it does not take, is the request's fault.
    std::string render_messages(const jmini::Value& body) {
        const std::vector<chat::Message> messages = messages_of(body);
        try { return format_.render(messages, true); }
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
        return params;
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
    // One streamed chunk of a compatible route: a chat delta or a text piece, with the finish reason on the last.
    std::string chunk(Route route, const std::string& id, const std::string& piece, bool first,
                      const std::string* finish) const {
        const std::string fr = finish ? jmini::quote(finish_reason(*finish)) : "null";
        if (route == Route::chat_completions) {
            std::string delta = first ? "{\"role\":\"assistant\",\"content\":" + jmini::quote(piece) + "}"
                              : finish ? "{}" : "{\"content\":" + jmini::quote(piece) + "}";
            return head(id, "chat.completion.chunk") + ",\"choices\":[{\"index\":0,\"delta\":" + delta +
                   ",\"finish_reason\":" + fr + "}]}";
        }
        return head(id, "text_completion") + ",\"choices\":[{\"index\":0,\"text\":" + jmini::quote(piece) +
               ",\"finish_reason\":" + fr + "}]}";
    }
    // The same chunk with a request's timings, for the one that carries the finish reason.
    std::string last_chunk(Route route, const std::string& id, const std::string& finish, const Request& r, size_t tokens) const {
        std::string c = chunk(route, id, "", false, &finish);
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
        try { r = sched_.submit(encode(prompt), params); }
        catch (const TooLong& e) { throw BadRequest(413, e.what()); }
        catch (const QueueFull& e) { throw BadRequest(503, e.what()); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }
        const std::string id = (route == Route::chat_completions ? "chatcmpl-" : "cmpl-") + std::to_string(next_id_.fetch_add(1));

        // Drain the channel.
        // A write that fails means the client went away: cancel the request and stop.
        // Between writes the socket is looked at every kProbe, token or not, since a request that is queued, prefilling or building a whole reply writes nothing that could fail.
        std::vector<uint32_t> gen;
        std::string text, pending;
        bool first = true;
        try {
            if (stream) c.begin_stream(200, "text/event-stream");
            uint32_t tid;
            Request::Clock::time_point probe = Request::Clock::now() + kProbe;
            for (;;) {
                const Request::Next got = r->next(tid, probe);
                if (got == Request::Next::end) break;
                if (Request::Clock::now() >= probe) {
                    if (c.peer_closed()) throw http::ClientGone("http: the client closed the connection");
                    probe = Request::Clock::now() + kProbe;
                }
                if (got == Request::Next::timeout) continue;
                gen.push_back(tid);
                pending += tok_.decode({tid});
                const size_t whole = utf8_complete(pending);
                const std::string piece = utf8_sanitize(pending.substr(0, whole));
                pending.erase(0, whole);
                text += piece;
                if (!stream) continue;
                if (compat(route)) c.write_chunk("data: " + chunk(route, id, piece, first, nullptr) + "\n\n");
                else c.write_chunk("data: {\"id\":" + std::to_string(tid) + ",\"text\":" + jmini::quote(piece) + "}\n\n");
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
                return;
            }
            const size_t prompt_tokens = r->prompt_tokens();
            if (stream && compat(route)) {
                if (!pending.empty() || first) c.write_chunk("data: " + chunk(route, id, pending, first, nullptr) + "\n\n");
                c.write_chunk("data: " + last_chunk(route, id, finish, *r, gen.size()) + "\n\n");
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
            } else if (route == Route::chat_completions) {
                c.respond(200, "application/json",
                          head(id, "chat.completion") + ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
                          jmini::quote(text) + "},\"finish_reason\":" + jmini::quote(finish_reason(finish)) + "}],\"usage\":" +
                          usage_json(prompt_tokens, gen.size()) + ",\"timings\":" + timings_json(*r, gen.size()) + "}");
            } else if (route == Route::completions) {
                c.respond(200, "application/json",
                          head(id, "text_completion") + ",\"choices\":[{\"index\":0,\"text\":" + jmini::quote(text) +
                          ",\"finish_reason\":" + jmini::quote(finish_reason(finish)) + "}],\"usage\":" +
                          usage_json(prompt_tokens, gen.size()) + ",\"timings\":" + timings_json(*r, gen.size()) + "}");
            } else {
                c.respond(200, "application/json",
                          "{\"text\":" + jmini::quote(text) + ",\"ids\":" + ids_json(gen) +
                          ",\"finish\":" + jmini::quote(finish) + ",\"prompt_tokens\":" +
                          std::to_string(prompt_tokens) + ",\"reused_tokens\":" + std::to_string(r->reused()) +
                          ",\"tokens\":" + std::to_string(gen.size()) + "}");
            }
        } catch (...) {
            r->cancel();
            uint32_t drop;
            while (r->next(drop, Request::Clock::now() + kProbe) != Request::Next::end) {}
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
    Scheduler sched(model, tok, cfg.max_seqs, cfg.max_queue);
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
