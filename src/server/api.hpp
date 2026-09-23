#pragma once
// The routes of docs/SERVER.md: native /v1/generate, /v1/chat and /v1/health, and the OpenAI-compatible /v1/chat/completions, /v1/completions and /v1/models.
// Both families share one parse, one scheduler request and one drain loop; one thread per connection parses, tokenizes, submits and drains.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>
#include <thread>
#include <vector>
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "inference/chat.hpp"
#include "server/http.hpp"
#include "server/scheduler.hpp"

namespace server {

struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    size_t max_seqs = 16;
    size_t max_queue = 64;   // requests waiting for admission; past it, 503
    size_t ubatch = 512;
    std::string model_name;
};

// The longest prefix of `bytes` that ends on a complete UTF-8 character, so a token whose text ends mid-character is held until the rest comes.
inline size_t utf8_complete(const std::string& bytes) {
    size_t i = bytes.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char)bytes[i - 1];
        if ((c & 0xC0) != 0x80) {
            const size_t need = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            return back + 1 >= need ? bytes.size() : i - 1;
        }
        --i;
        ++back;
    }
    return bytes.size();
}

// The text with every byte sequence that is not valid UTF-8 replaced by U+FFFD, since a byte-level vocabulary can sample bytes that form no character and JSON carries only characters.
inline std::string utf8_sanitize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t len = c < 0x80 ? 1 : (c >= 0xC2 && c < 0xE0) ? 2 : (c >= 0xE0 && c < 0xF0) ? 3 : (c >= 0xF0 && c < 0xF5) ? 4 : 0;
        bool ok = len != 0 && i + len <= s.size();
        for (size_t j = 1; ok && j < len; ++j)
            if (((unsigned char)s[i + j] & 0xC0) != 0x80) ok = false;
        if (ok) { out.append(s, i, len); i += len; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

class Api {
public:
    Api(infer::Model& model, const bpe::Tokenizer& tok, const gguf::GGUFModel& file, Scheduler& sched,
        const Config& cfg)
        : model_(model), tok_(tok), sched_(sched), cfg_(cfg), started_((int64_t)std::time(nullptr)) {
        template_ = chat::get_chat_template(file);
        if (template_.empty())
            template_ = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                        "{{ message['content'] }}<|im_end|>\n{% endfor %}"
                        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        bos_ = tok.bos_id >= 0 && (size_t)tok.bos_id < tok.vocab.size() ? tok.vocab[(size_t)tok.bos_id] : "";
        eos_ = tok.eos_id >= 0 && (size_t)tok.eos_id < tok.vocab.size() ? tok.vocab[(size_t)tok.eos_id] : "";
    }

    void handle(http::Connection& c) {
        http::Request req;
        int status = 0;
        http::Limits limits;
        if (!c.read_request(req, limits, status)) {
            if (status) c.respond(status, "application/json", error_json(http::reason(status), false));
            return;
        }
        // The compatible routes report errors in their clients' shape.
        const bool compat = req.path == "/v1/chat/completions" || req.path == "/v1/completions";
        try {
            if (req.method == "GET" && req.path == "/v1/health") return health(c);
            if (req.method == "GET" && req.path == "/v1/models") return models(c);
            if (req.method == "POST" && req.path == "/v1/generate") return generate(c, req, Route::generate);
            if (req.method == "POST" && req.path == "/v1/chat") return generate(c, req, Route::chat);
            if (req.method == "POST" && req.path == "/v1/chat/completions") return generate(c, req, Route::chat_completions);
            if (req.method == "POST" && req.path == "/v1/completions") return generate(c, req, Route::completions);
            c.respond(404, "application/json", error_json("no such route", compat));
        } catch (const BadRequest& e) {
            c.respond(e.status, "application/json", error_json(e.what(), compat));
        } catch (const std::exception& e) {
            try { c.respond(500, "application/json", error_json(e.what(), compat)); } catch (...) {}
        }
    }

private:
    enum class Route { generate, chat, chat_completions, completions };

    struct BadRequest : std::runtime_error {
        int status;
        BadRequest(int s, const std::string& m) : std::runtime_error(m), status(s) {}
    };
    static std::string error_json(const std::string& message, bool compat) {
        if (!compat) return "{\"error\":" + jmini::quote(message) + "}";
        return "{\"error\":{\"message\":" + jmini::quote(message) + ",\"type\":\"invalid_request_error\"}}";
    }

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

    // The cap a compatible request gets when it sends none; replaced by the room left once its prompt is encoded.
    static constexpr int kUntilLimit = -1;
    static double number(const jmini::Value& v, const char* key, double fallback) {
        const jmini::Value* f = v.get(key);
        if (!f) return fallback;
        if (!f->isNumber() || !std::isfinite(f->asNumber())) throw BadRequest(400, std::string(key) + " must be a number");
        return f->asNumber();
    }
    static bool flag(const jmini::Value& v, const char* key) {
        const jmini::Value* f = v.get(key);
        return f && f->t == jmini::Value::T::Bool && f->b;
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

    std::string render_messages(const jmini::Value& body) {
        const jmini::Value* msgs = body.get("messages");
        if (!msgs || !msgs->isArray() || msgs->asArray().empty()) throw BadRequest(400, "messages must be a non-empty array");
        std::vector<chat::Message> messages;
        for (const auto& m : msgs->asArray()) {
            const jmini::Value* role = m.get("role");
            if (!role || !role->isString()) throw BadRequest(400, "every message needs a string role");
            messages.push_back({role->asString(), content_of(m)});
        }
        return chat::render(template_, messages, true, bos_, eos_);
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
        const bool compat = route == Route::chat_completions || route == Route::completions;
        SampleParams params;
        // The compatible routes take an absent cap as the standard does, no cap: the reply runs to the model's end of text or to what the request may hold, set once the prompt is encoded (kUntilLimit). The native route keeps its 64.
        params.max_tokens = (int)number(body, "max_tokens", compat ? number(body, "max_completion_tokens", kUntilLimit) : 64);
        params.temp = (float)number(body, "temperature", 0.8);
        params.top_k = (int)number(body, "top_k", 40);
        params.top_p = (float)number(body, "top_p", 0.95);
        params.penalty = (float)number(body, "penalty", compat ? number(body, "repetition_penalty", 1.0) : 1.0);
        params.seed = (uint64_t)number(body, "seed", 0);
        if (const jmini::Value* stop = body.get("stop")) {
            if (stop->isString()) params.stop.push_back(stop->asString());
            else if (stop->isArray())
                for (const auto& s : stop->asArray())
                    if (s.isString()) params.stop.push_back(s.asString());
        }
        if (compat && number(body, "n", 1) != 1) throw BadRequest(400, "n must be 1");
        return params;
    }

    static std::string finish_reason(const std::string& finish) {
        return finish == "length" ? "length" : "stop";
    }
    std::string usage_json(size_t prompt_tokens, size_t tokens) const {
        return "{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(tokens) +
               ",\"total_tokens\":" + std::to_string(prompt_tokens + tokens) + "}";
    }
    // A finished request's speed in the fields clients such as Open WebUI read beside the standard usage: prompt tokens prefilled and reused, milliseconds to the first token, and generation after it.
    static std::string timings_json(const Request& r, size_t tokens) {
        const Request::Timings t = r.timings();
        const size_t cached = r.reused(), prefilled = r.prompt_tokens() - cached, predicted = tokens > 1 ? tokens - 1 : 0;
        auto rate = [](size_t n, double ms) { return ms > 0 ? 1000.0 * (double)n / ms : 0.0; };
        auto num = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.3f", v); return std::string(b); };
        return "{\"cache_n\":" + std::to_string(cached) + ",\"prompt_n\":" + std::to_string(prefilled) +
               ",\"prompt_ms\":" + num(t.prompt_ms) + ",\"prompt_per_second\":" + num(rate(prefilled, t.prompt_ms)) +
               ",\"predicted_n\":" + std::to_string(predicted) + ",\"predicted_ms\":" + num(t.predicted_ms) +
               ",\"predicted_per_second\":" + num(rate(predicted, t.predicted_ms)) + ",\"queued_ms\":" + num(t.queued_ms) + "}";
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

    void generate(http::Connection& c, const http::Request& req, Route route) {
        jmini::Value body;
        try { body = jmini::parse(req.body); }
        catch (const std::exception& e) { throw BadRequest(400, std::string("bad JSON: ") + e.what()); }
        if (!body.isObject()) throw BadRequest(400, "the body must be a JSON object");
        const bool compat = route == Route::chat_completions || route == Route::completions;

        const std::string prompt = prompt_of(body, route);
        SampleParams params = params_of(body, route);
        const bool stream = flag(body, "stream");
        const jmini::Value* so = body.get("stream_options");
        const bool include_usage = so && so->isObject() && flag(*so, "include_usage");

        std::vector<uint32_t> ids = tok_.encode(prompt);
        if (ids.empty()) throw BadRequest(400, "the prompt encodes to no tokens");
        // An uncapped request may run to its token limit; the scheduler reserves its blocks as it grows, so uncapped requests run side by side.
        if (params.max_tokens == kUntilLimit) {
            if (ids.size() >= sched_.token_limit())
                throw BadRequest(413, "the prompt fills the " + std::to_string(sched_.token_limit()) + " tokens a request may hold");
            params.max_tokens = (int)(sched_.token_limit() - ids.size());
            params.until_limit = true;
        }
        if (ids.size() + (size_t)std::max(params.max_tokens, 0) > sched_.token_limit())
            throw BadRequest(413, "prompt plus max_tokens exceeds the " + std::to_string(sched_.token_limit()) +
                                      " tokens a request may hold");
        std::shared_ptr<Request> r;
        try { r = sched_.submit(std::move(ids), params); }
        catch (const QueueFull& e) { throw BadRequest(503, e.what()); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }
        const std::string id = (route == Route::chat_completions ? "chatcmpl-" : "cmpl-") + std::to_string(next_id_.fetch_add(1));

        // Drain the channel.
        // A write that fails means the client went away: cancel the request and stop.
        std::vector<uint32_t> gen;
        std::string text, pending;
        bool first = true;
        try {
            if (stream) c.begin_stream(200, "text/event-stream");
            uint32_t tid;
            while (r->next(tid)) {
                gen.push_back(tid);
                pending += tok_.decode({tid});
                const size_t whole = utf8_complete(pending);
                const std::string piece = utf8_sanitize(pending.substr(0, whole));
                pending.erase(0, whole);
                text += piece;
                if (!stream) continue;
                if (compat) c.write_chunk("data: " + chunk(route, id, piece, first, nullptr) + "\n\n");
                else c.write_chunk("data: {\"id\":" + std::to_string(tid) + ",\"text\":" + jmini::quote(piece) + "}\n\n");
                first = false;
            }
            // Whatever is left never completed a character.
            pending = utf8_sanitize(pending);
            text += pending;
            const std::string finish = r->finish();
            if (finish == "error") throw std::runtime_error(r->error());
            const size_t prompt_tokens = r->prompt_tokens();
            if (stream && compat) {
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
                std::string ids_json = "[";
                for (size_t i = 0; i < gen.size(); ++i) ids_json += (i ? "," : "") + std::to_string(gen[i]);
                ids_json += "]";
                c.respond(200, "application/json",
                          "{\"text\":" + jmini::quote(text) + ",\"ids\":" + ids_json +
                          ",\"finish\":" + jmini::quote(finish) + ",\"prompt_tokens\":" +
                          std::to_string(prompt_tokens) + ",\"reused_tokens\":" + std::to_string(r->reused()) +
                          ",\"tokens\":" + std::to_string(gen.size()) + "}");
            }
        } catch (...) {
            r->cancel();
            uint32_t drop;
            while (r->next(drop)) {}
            throw;
        }
    }

    infer::Model& model_;
    const bpe::Tokenizer& tok_;
    Scheduler& sched_;
    Config cfg_;
    const int64_t started_;
    std::atomic<uint64_t> next_id_{1};
    std::string template_, bos_, eos_;
};

// Serve until the listener is closed: the scheduler on its own thread, the accept loop here, one detached thread per connection.
inline void serve(infer::Model& model, const bpe::Tokenizer& tok, const gguf::GGUFModel& file,
                  const Config& cfg, http::Listener& listener) {
    Scheduler sched(model, tok, cfg.max_seqs, cfg.ubatch, cfg.max_queue);
    Api api(model, tok, file, sched, cfg);
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
