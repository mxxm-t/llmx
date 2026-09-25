// The HTTP layer of docs/SERVER.md step 1: a listener on a system-chosen port served from a thread, requests sent with the layer's own client.
// A whole response, a body echoed back, a chunked stream whose chunks arrive as written, a whole response refused inside a stream, an oversized body refused with 413, a malformed request line refused with 400, an unknown route 404, and the listener closed from the main thread ending the accept loop.
#include <atomic>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "server/http.hpp"

namespace {
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

void serve_one(http::Connection& c) {
    http::Request req;
    int status = 0;
    http::Limits limits;
    limits.max_body_bytes = 1024;
    if (!c.read_request(req, limits, status)) {
        if (status) c.respond(status, "text/plain", http::reason(status));
        return;
    }
    if (req.method == "GET" && req.path == "/ping") {
        c.respond(200, "text/plain", "pong " + req.query);
    } else if (req.method == "POST" && req.path == "/echo") {
        c.respond(200, req.headers["content-type"], req.body);
    } else if (req.method == "GET" && req.path == "/stream") {
        c.begin_stream(200, "text/event-stream");
        // Three chunks; the second and third split a two-byte character's neighbourhood only at chunk edges, never inside a character.
        c.write_chunk("data: one\n\n");
        c.write_chunk("data: tw\xc3\xb6\n\n");
        c.write_chunk("data: [DONE]\n\n");
        c.end_stream();
    } else if (req.method == "GET" && req.path == "/respond-in-stream") {
        // A second head would land inside the chunked body, so respond refuses and the stream carries on.
        c.begin_stream(200, "text/plain");
        bool refused = false;
        try { c.respond(500, "text/plain", "late"); } catch (const std::logic_error&) { refused = true; }
        c.write_chunk(refused ? "refused" : "sent");
        c.end_stream();
    } else {
        c.respond(404, "text/plain", "no such route");
    }
}
}

int main() {
    try {
        http::Listener listener("127.0.0.1", 0);
        const uint16_t port = listener.port();
        require(port != 0, "no port assigned");
        std::atomic<int> served{0};
        std::thread server([&] {
            for (;;) {
                http::Connection c = listener.accept();
                if (!c.open()) break;
                try { serve_one(c); } catch (const std::exception&) {}
                ++served;
            }
        });

        std::string out;
        require(http::fetch("127.0.0.1", port, "GET", "/ping?x=1", "", out) == 200 && out == "pong x=1",
                "GET /ping");
        require(http::fetch("127.0.0.1", port, "POST", "/echo", "{\"a\":[1,2,3]}", out) == 200 &&
                out == "{\"a\":[1,2,3]}", "POST /echo");
        std::vector<std::string> chunks;
        require(http::fetch("127.0.0.1", port, "GET", "/stream", "", out, &chunks) == 200, "GET /stream");
        require(chunks.size() == 3 && chunks[1] == "data: tw\xc3\xb6\n\n" &&
                out == "data: one\n\ndata: tw\xc3\xb6\n\ndata: [DONE]\n\n", "chunks as written");
        require(http::fetch("127.0.0.1", port, "GET", "/respond-in-stream", "", out) == 200 && out == "refused",
                "a whole response refused inside a stream");
        require(http::fetch("127.0.0.1", port, "POST", "/echo", std::string(2048, 'x'), out) == 413,
                "oversized body refused");
        require(http::fetch("127.0.0.1", port, "GET", "/nowhere", "", out) == 404, "unknown route");
        {
            // A malformed request line, by hand.
            http::platform_init();
            const http::Socket s = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
            require(::connect(s, (const sockaddr*)&addr, sizeof(addr)) == 0, "connect");
            const std::string bad = "NOTHTTP\r\n\r\n";
            ::send(s, bad.data(), (int)bad.size(), 0);
            std::string raw;
            char buf[512];
            for (;;) {
                const int n = (int)::recv(s, buf, sizeof(buf), 0);
                if (n <= 0) break;
                raw.append(buf, (size_t)n);
            }
            http::close_socket(s);
            require(raw.rfind("HTTP/1.1 400", 0) == 0, "malformed request refused with 400");
        }
        listener.close();
        server.join();
        require(served.load() == 7, "every request served once");
        std::cout << "http: listener, whole responses, a 3-chunk stream, a response refused inside a stream, 413, 400 and 404 over " << served.load()
                  << " connections\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "http: " << e.what() << "\n";
        return 1;
    }
}
