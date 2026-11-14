// The HTTP layer of docs/SERVER.md step 1: a listener on a system-chosen port served from a thread, requests sent with the layer's own client (docs/TESTS.md).
// The hosted TSan job runs it under ThreadSanitizer, which sees the socket read by accept on one thread and closed on another.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "server/http.hpp"

namespace http {

// The client side: one request, the whole response read to the end, chunked bodies decoded.
// Returns the status; `chunks` receives each chunk as it arrived when the reply was chunked, so a test can see where the server split the stream.
int fetch(const std::string& host, uint16_t port, const std::string& method, const std::string& path,
                 const std::string& body, std::string& out, std::vector<std::string>* chunks = nullptr,
                 const std::string& content_type = "application/json") {
    platform_init();
    const Socket s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == kInvalid) throw std::runtime_error("http: socket failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (::connect(s, (const sockaddr*)&addr, sizeof(addr)) != 0) {
        close_socket(s);
        throw std::runtime_error("http: connect failed");
    }
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n";
    if (!body.empty()) req += "Content-Type: " + content_type + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    size_t off = 0;
    while (off < req.size()) {
        const int n = (int)::send(s, req.data() + off, (int)(req.size() - off), kSendFlags);
        if (n <= 0) { close_socket(s); throw std::runtime_error("http: send failed"); }
        off += (size_t)n;
    }
    std::string raw;
    char buf[16384];
    for (;;) {
        const int n = (int)::recv(s, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, (size_t)n);
    }
    close_socket(s);
    const size_t head_end = raw.find("\r\n\r\n");
    if (head_end == std::string::npos) throw std::runtime_error("http: no response head");
    const std::string head = raw.substr(0, head_end);
    const int status = std::atoi(head.c_str() + 9);
    std::string lower = head;
    for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
    const std::string payload = raw.substr(head_end + 4);
    out.clear();
    if (lower.find("transfer-encoding: chunked") != std::string::npos) {
        size_t pos = 0;
        for (;;) {
            const size_t line_end = payload.find("\r\n", pos);
            if (line_end == std::string::npos) throw std::runtime_error("http: truncated chunked body");
            const size_t size = std::strtoull(payload.substr(pos, line_end - pos).c_str(), nullptr, 16);
            pos = line_end + 2;
            if (size == 0) break;
            if (pos + size > payload.size()) throw std::runtime_error("http: truncated chunk");
            const std::string chunk = payload.substr(pos, size);
            if (chunks) chunks->push_back(chunk);
            out += chunk;
            pos += size + 2;
        }
    } else {
        out = payload;
    }
    return status;
}

} // namespace http

namespace {
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

std::atomic<int> departures{0};
// Clients that stopped reading while the server streamed to them, each seen as gone within the send timeout.
std::atomic<int> drowned{0};

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
    // Short deadlines, so a client that sends nothing, half a body or reads nothing is refused within the test's seconds.
    limits.head_ms = std::chrono::milliseconds(300);
    limits.body_ms = std::chrono::milliseconds(300);
    limits.body_ms_per_mib = std::chrono::milliseconds(0);
    limits.send_ms = std::chrono::milliseconds(300);
    if (!c.read_head(req, limits, status) || !c.read_body(req, limits, status)) {
        if (status) {
            c.respond(status, "text/plain", http::reason(status));
            c.linger();
        }
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
    } else if (req.method == "GET" && req.path == "/drown") {
        // The client reads nothing: once its buffers are full a write must fail within the send timeout rather than block for ever, and throw ClientGone.
        c.begin_stream(200, "text/plain");
        const std::string chunk(64 * 1024, 'd');
        const auto began = std::chrono::steady_clock::now();
        for (size_t written = 0; written < size_t(256) << 20; written += chunk.size()) {
            try { c.write_chunk(chunk); } catch (const http::ClientGone&) { ++drowned; break; }
            require(std::chrono::steady_clock::now() - began < std::chrono::seconds(10), "a write to a client that reads nothing did not fail");
        }
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
            // A body past the limit, sent whole with its head before the client reads, as a client library sends: the 413 must arrive, which a close with the body unread would turn into a reset.
            const http::Socket s = connect_to(port);
            const std::string big = "POST /echo HTTP/1.1\r\nContent-Length: 1048576\r\n\r\n" + std::string(1 << 20, 'y');
            bool sent = true;
            for (size_t off = 0; sent && off < big.size();) {
                const int n = (int)::send(s, big.data() + off, (int)(big.size() - off), 0);
                if (n <= 0) sent = false;
                else off += (size_t)n;
            }
            std::string raw;
            char buf[512];
            for (;;) {
                const int n = (int)::recv(s, buf, sizeof(buf), 0);
                if (n <= 0) break;
                raw.append(buf, (size_t)n);
            }
            http::close_socket(s);
            require(sent && raw.rfind("HTTP/1.1 413", 0) == 0, "a body past the limit sent whole was not answered 413");
        }
        {
            // A refusal from a thread that cannot wait, handed to Lingering before the client has sent: the request then meets an open socket, so the client reads the 503 and an end of stream rather than a reset.
            http::Listener alone("127.0.0.1", 0);
            http::Lingering lingering(4);
            const http::Socket s = connect_to(alone.port());
            http::Connection c = alone.accept();
            require(c.open(), "the connection to refuse was not accepted");
            c.respond(503, "text/plain", "full");
            lingering.add(std::move(c), std::chrono::milliseconds(300));
            const std::string req = "GET /ping HTTP/1.1\r\n\r\n";
            require(::send(s, req.data(), (int)req.size(), 0) == (int)req.size(), "the request after the refusal was not sent");
            std::string raw;
            char buf[512];
            int n;
            while ((n = (int)::recv(s, buf, sizeof(buf), 0)) > 0) raw.append(buf, (size_t)n);
            http::close_socket(s);
            require(n == 0 && raw.rfind("HTTP/1.1 503", 0) == 0, "a refusal handed to Lingering was not read as 503 and an end of stream");
        }
        {
            // Lingering holds at most its limit: a refusal past it closes the oldest at once, so refused clients that hold their sockets cost no more than the limit's descriptors, while the newest still drains to its end of stream.
            http::Listener alone("127.0.0.1", 0);
            http::Lingering lingering(2);
            http::Socket clients[3];
            for (http::Socket& s : clients) {
                s = connect_to(alone.port());
                http::Connection c = alone.accept();
                require(c.open(), "a connection for the capped Lingering was not accepted");
                c.respond(503, "text/plain", "full");
                lingering.add(std::move(c), std::chrono::seconds(10));
            }
            const auto t0 = std::chrono::steady_clock::now();
            std::string raw;
            char buf[512];
            int n;
            while ((n = (int)::recv(clients[0], buf, sizeof(buf), 0)) > 0) raw.append(buf, (size_t)n);
            require(n == 0 && raw.rfind("HTTP/1.1 503", 0) == 0, "the oldest refusal past Lingering's limit was not read as 503 and an end of stream");
            require(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5), "the oldest refusal past Lingering's limit was not closed at once");
            const std::string req = "GET /ping HTTP/1.1\r\n\r\n";
            require(::send(clients[2], req.data(), (int)req.size(), 0) == (int)req.size(), "the request after the capped refusal was not sent");
            ::shutdown(clients[2], http::kShutdownSend);
            raw.clear();
            while ((n = (int)::recv(clients[2], buf, sizeof(buf), 0)) > 0) raw.append(buf, (size_t)n);
            require(n == 0 && raw.rfind("HTTP/1.1 503", 0) == 0, "the newest refusal in a capped Lingering was not read as 503 and an end of stream");
            for (http::Socket s : clients) http::close_socket(s);
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
        // A client that sends nothing, and one that sends a head and only part of the body it announces: each is answered 408 within the head or body deadline and the connection closed.
        for (const char* partial : {"", "POST /echo HTTP/1.1\r\nContent-Length: 100\r\n\r\nten bytes!"}) {
            const http::Socket s = connect_to(port);
            if (*partial) ::send(s, partial, (int)std::strlen(partial), 0);
            const auto began = std::chrono::steady_clock::now();
            std::string raw;
            char buf[512];
            for (;;) {
                const int n = (int)::recv(s, buf, sizeof(buf), 0);
                if (n <= 0) break;
                raw.append(buf, (size_t)n);
            }
            http::close_socket(s);
            require(raw.rfind("HTTP/1.1 408", 0) == 0, *partial ? "a half-sent body answered 408" : "a client that sent nothing answered 408");
            require(std::chrono::steady_clock::now() - began < std::chrono::seconds(5), "the 408 did not come within the deadline");
        }
        {
            // A client that reads nothing of its stream: the server's write fails within the send timeout and the client leaves afterwards.
            const http::Socket s = connect_to(port);
            const std::string get = "GET /drown HTTP/1.1\r\n\r\n";
            ::send(s, get.data(), (int)get.size(), 0);
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (drowned.load() == 0) {
                require(std::chrono::steady_clock::now() < until, "a client that reads nothing was not seen as gone");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            http::close_socket(s);
        }
        // Closed while the server thread may still be serving /linger, so its next accept can start after the close.
        listener.close();
        server.join();
        require(served.load() == 13, "every request served once");
        require(departures.load() == 1, "a departed client seen as closed and a write to it refused");
        require(drowned.load() == 1, "a client that reads nothing seen as gone once");
        {
            // A listener closed while its thread waits in accept, as a server with no client stops: that accept returns no connection within seconds, and so does any accept after it.
            http::Listener idle("127.0.0.1", 0);
            std::atomic<bool> started{false}, empty{false};
            std::thread waiter([&] {
                started = true;
                empty = !idle.accept().open();
            });
            while (!started.load()) std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const auto closed_at = std::chrono::steady_clock::now();
            idle.close();
            waiter.join();
            require(empty.load(), "an accept waiting when its listener closed gave a connection");
            require(std::chrono::steady_clock::now() - closed_at < std::chrono::seconds(5), "an accept waiting when its listener closed did not return");
            require(!idle.accept().open(), "an accept on a closed listener gave a connection");
        }
        std::cout << "http: listener, whole responses, a 3-chunk stream, a response refused inside a stream, 413 (also for a body sent whole), 400 and 404, a client with an urgent byte answered, a client seen leaving, "
                  << "408 for a silent client and a half-sent body, a client that reads nothing dropped, a refusal handed to Lingering read as 503 and an end of stream, a refusal past Lingering's limit closing the oldest at once, over " << served.load()
                  << " connections, and a listener closed while serving and while waiting\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "http: " << e.what() << "\n";
        return 1;
    }
}
