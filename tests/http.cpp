// The HTTP layer of docs/SERVER.md step 1: a listener on a system-chosen port served from a thread, requests sent with the layer's own client.
// A whole response, a body echoed back, a chunked stream whose chunks arrive as written, a whole response refused inside a stream, an oversized body refused with 413, a malformed request line refused with 400, an unknown route 404, a client seen as open while it waits, also after one urgent (out-of-band) byte, and as closed once it leaves, a write to it then throwing ClientGone, and the listener closed from the main thread ending the accept loop.
#include <atomic>
#include <chrono>
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

std::atomic<int> departures{0};

http::Socket connect_to(uint16_t port) {
    http::platform_init();
    const http::Socket s = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    require(::connect(s, (const sockaddr*)&addr, sizeof(addr)) == 0, "connect");
    return s;
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
    } else if (req.method == "GET" && req.path == "/urgent") {
        // The client sent one urgent byte after its request and waits; the look at it must neither take that byte for normal data and wait for more nor take the client as gone.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        c.respond(200, "text/plain", c.peer_closed() ? "closed" : "open");
    } else if (req.method == "GET" && req.path == "/linger") {
        // The client waits for this answer, so it is open here; then it leaves, which the connection must see without reading.
        c.respond(200, "text/plain", c.peer_closed() ? "closed" : "open");
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!c.peer_closed()) {
            require(std::chrono::steady_clock::now() < until, "a departed client not seen");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        // A write to it fails once the client's end has refused what the first write sent, and throws ClientGone, which a server answers with nothing.
        for (;;) {
            try { c.write_chunk("late"); } catch (const http::ClientGone&) { break; }
            require(std::chrono::steady_clock::now() < until, "a write to a departed client did not fail");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ++departures;
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
            const http::Socket s = connect_to(port);
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
        {
            // A client that sends one urgent byte after its request and waits a few seconds at most for the answer, which must come at once and say open.
            const http::Socket s = connect_to(port);
            int one = 1;
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
#if defined(_WIN32)
            const DWORD wait = 5000;
#else
            const timeval wait{5, 0};
#endif
            setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&wait, sizeof(wait));
            const std::string get = "GET /urgent HTTP/1.1\r\n\r\n";
            ::send(s, get.data(), (int)get.size(), 0);
            ::send(s, "!", 1, MSG_OOB);
            std::string raw;
            char buf[512];
            while (raw.find("\r\n\r\n") == std::string::npos || raw.size() < raw.find("\r\n\r\n") + 8) {
                const int n = (int)::recv(s, buf, sizeof(buf), 0);
                if (n <= 0) break;
                raw.append(buf, (size_t)n);
            }
            http::close_socket(s);
            const size_t head = raw.find("\r\n\r\n");
            require(head != std::string::npos && raw.substr(head + 4, 4) == "open",
                    "a client that sent an urgent byte answered at once and seen as open");
        }
        {
            // A client that reads its answer without closing, then leaves; the server waits for the end of its stream rather than reading to it.
            const http::Socket s = connect_to(port);
            const std::string get = "GET /linger HTTP/1.1\r\n\r\n";
            ::send(s, get.data(), (int)get.size(), 0);
            std::string raw;
            char buf[512];
            while (raw.find("\r\n\r\n") == std::string::npos || raw.size() < raw.find("\r\n\r\n") + 8) {
                const int n = (int)::recv(s, buf, sizeof(buf), 0);
                require(n > 0, "the answer to /linger");
                raw.append(buf, (size_t)n);
            }
            http::close_socket(s);
            require(raw.substr(raw.find("\r\n\r\n") + 4, 4) == "open", "a waiting client seen as open");
        }
        listener.close();
        server.join();
        require(served.load() == 9, "every request served once");
        require(departures.load() == 1, "a departed client seen as closed and a write to it refused");
        std::cout << "http: listener, whole responses, a 3-chunk stream, a response refused inside a stream, 413, 400 and 404, a client with an urgent byte answered, a client seen leaving, over " << served.load()
                  << " connections\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "http: " << e.what() << "\n";
        return 1;
    }
}
