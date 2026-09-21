#pragma once
// The routes of docs/SERVER.md over the HTTP layer and the scheduler:
// /v1/generate and /v1/chat, streamed as server-sent events or returned
// whole, /v1/health and /v1/models. One thread per connection parses,
// tokenizes, submits and drains; the scheduler thread runs the model.
#include <atomic>
#include <cmath>
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
    size_t ubatch = 512;
    std::string model_name;
};

// The longest prefix of `bytes` that ends on a complete UTF-8 character,
// so a token whose text ends mid-character is held until the rest comes.
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

// The text with every byte sequence that is not valid UTF-8 replaced by
// U+FFFD, since a byte-level vocabulary can sample bytes that form no
// character and JSON carries only characters.
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
        : model_(model), tok_(tok), sched_(sched), cfg_(cfg) {
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
            if (status) c.respond(status, "application/json", error_json(http::reason(status)));
            return;
        }
        try {
            if (req.method == "GET" && req.path == "/v1/health") return health(c);
            if (req.method == "GET" && req.path == "/v1/models") return models(c);
            if (req.method == "POST" && (req.path == "/v1/generate" || req.path == "/v1/chat"))
                return generate(c, req, req.path == "/v1/chat");
            c.respond(404, "application/json", error_json("no such route"));
        } catch (const BadRequest& e) {
            c.respond(e.status, "application/json", error_json(e.what()));
        } catch (const std::exception& e) {
            try { c.respond(500, "application/json", error_json(e.what())); } catch (...) {}
        }
    }

private:
    struct BadRequest : std::runtime_error {
        int status;
        BadRequest(int s, const std::string& m) : std::runtime_error(m), status(s) {}
    };
    static std::string error_json(const std::string& message) {
        return "{\"error\":" + jmini::quote(message) + "}";
    }

    void health(http::Connection& c) {
        const Scheduler::Stats s = sched_.stats();
        c.respond(200, "application/json",
                  "{\"status\":\"ok\",\"model\":" + jmini::quote(cfg_.model_name) +
                  ",\"active\":" + std::to_string(s.active) + ",\"queued\":" + std::to_string(s.queued) + "}");
    }
    void models(http::Connection& c) {
        c.respond(200, "application/json",
                  "{\"models\":[{\"name\":" + jmini::quote(cfg_.model_name) +
                  ",\"context_length\":" + std::to_string(model_.config().context_length) +
                  ",\"vocab\":" + std::to_string(model_.n_vocab()) + "}]}");
    }

    static double number(const jmini::Value& v, const char* key, double fallback) {
        const jmini::Value* f = v.get(key);
        if (!f) return fallback;
        if (!f->isNumber() || !std::isfinite(f->asNumber())) throw BadRequest(400, std::string(key) + " must be a number");
        return f->asNumber();
    }

    void generate(http::Connection& c, const http::Request& req, bool chat_route) {
        jmini::Value body;
        try { body = jmini::parse(req.body); }
        catch (const std::exception& e) { throw BadRequest(400, std::string("bad JSON: ") + e.what()); }
        if (!body.isObject()) throw BadRequest(400, "the body must be a JSON object");

        std::string prompt;
        if (chat_route) {
            const jmini::Value* msgs = body.get("messages");
            if (!msgs || !msgs->isArray() || msgs->asArray().empty()) throw BadRequest(400, "messages must be a non-empty array");
            std::vector<chat::Message> messages;
            for (const auto& m : msgs->asArray()) {
                const jmini::Value* role = m.get("role");
                const jmini::Value* content = m.get("content");
                if (!role || !role->isString() || !content || !content->isString())
                    throw BadRequest(400, "every message needs a string role and content");
                messages.push_back({role->asString(), content->asString()});
            }
            prompt = chat::render(template_, messages, true, bos_, eos_);
        } else {
            const jmini::Value* p = body.get("prompt");
            if (!p || !p->isString() || p->asString().empty()) throw BadRequest(400, "prompt must be a non-empty string");
            prompt = p->asString();
        }
        SampleParams params;
        params.max_tokens = (int)number(body, "max_tokens", 64);
        params.temp = (float)number(body, "temperature", 0.8);
        params.top_k = (int)number(body, "top_k", 40);
        params.top_p = (float)number(body, "top_p", 0.95);
        params.penalty = (float)number(body, "penalty", 1.0);
        params.seed = (uint64_t)number(body, "seed", 0);
        if (const jmini::Value* stop = body.get("stop")) {
            if (stop->isString()) params.stop.push_back(stop->asString());
            else if (stop->isArray())
                for (const auto& s : stop->asArray())
                    if (s.isString()) params.stop.push_back(s.asString());
        }
        bool stream = false;
        if (const jmini::Value* s = body.get("stream")) stream = s->t == jmini::Value::T::Bool && s->b;

        std::vector<uint32_t> ids = tok_.encode(prompt);
        if (ids.empty()) throw BadRequest(400, "the prompt encodes to no tokens");
        if (ids.size() + (size_t)std::max(params.max_tokens, 0) > (size_t)model_.config().context_length)
            throw BadRequest(413, "prompt plus max_tokens exceeds the context of " +
                                      std::to_string(model_.config().context_length) + " tokens");
        std::shared_ptr<Request> r;
        try { r = sched_.submit(std::move(ids), params); }
        catch (const std::exception& e) { throw BadRequest(400, e.what()); }

        // Drain the channel. A write that fails means the client went
        // away: cancel the request and stop.
        std::vector<uint32_t> gen;
        std::string text, pending;
        try {
            if (stream) c.begin_stream(200, "text/event-stream");
            uint32_t id;
            while (r->next(id)) {
                gen.push_back(id);
                pending += tok_.decode({id});
                const size_t whole = utf8_complete(pending);
                const std::string piece = utf8_sanitize(pending.substr(0, whole));
                pending.erase(0, whole);
                text += piece;
                if (stream)
                    c.write_chunk("data: {\"id\":" + std::to_string(id) + ",\"text\":" + jmini::quote(piece) + "}\n\n");
            }
            // Whatever is left never completed a character.
            pending = utf8_sanitize(pending);
            text += pending;
            const std::string finish = r->finish();
            if (finish == "error") throw std::runtime_error(r->error());
            if (stream) {
                if (!pending.empty()) c.write_chunk("data: {\"text\":" + jmini::quote(pending) + "}\n\n");
                c.write_chunk("data: {\"done\":true,\"finish\":" + jmini::quote(finish) +
                              ",\"tokens\":" + std::to_string(gen.size()) + "}\n\ndata: [DONE]\n\n");
                c.end_stream();
            } else {
                std::string ids_json = "[";
                for (size_t i = 0; i < gen.size(); ++i) ids_json += (i ? "," : "") + std::to_string(gen[i]);
                ids_json += "]";
                c.respond(200, "application/json",
                          "{\"text\":" + jmini::quote(text) + ",\"ids\":" + ids_json +
                          ",\"finish\":" + jmini::quote(finish) + ",\"prompt_tokens\":" +
                          std::to_string(r->prompt().size()) + ",\"tokens\":" + std::to_string(gen.size()) + "}");
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
    std::string template_, bos_, eos_;
};

// Serve until the listener is closed: the scheduler on its own thread,
// the accept loop here, one detached thread per connection.
inline void serve(infer::Model& model, const bpe::Tokenizer& tok, const gguf::GGUFModel& file,
                  const Config& cfg, http::Listener& listener) {
    Scheduler sched(model, tok, cfg.max_seqs, cfg.ubatch);
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
