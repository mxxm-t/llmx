#pragma once
// HTTP/1.1 over blocking sockets, enough for the server in docs/SERVER.md: listen, accept, read one request with a Content-Length body, write one response or a chunked stream, and tell whether the client has left.
// One thread per connection, no keep-alive beyond one request, no TLS, no external library: a reverse proxy does the rest when the server faces a network.
// Windows uses Winsock, everything else BSD sockets; the two differ only in the handle type, in how a socket is closed or polled and in their error codes, which is what the few #if blocks below cover.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace http {

#if defined(_WIN32)
using Socket = SOCKET;
constexpr Socket kInvalid = INVALID_SOCKET;
inline void close_socket(Socket s) { closesocket(s); }
inline int last_error() { return WSAGetLastError(); }
// The accept failures that belong to one client, a connection reset before it was taken or an interrupted call, after which the next accept can succeed at once.
inline bool per_client(int e) { return e == WSAECONNRESET || e == WSAEINTR; }
// A failed read that says nothing about the connection: no data yet, or an interrupted call.
inline bool transient(int e) { return e == WSAEWOULDBLOCK || e == WSAEINTR; }
// Whether a read would return at once, with data, an end of stream or an error, asked without waiting.
// Normal data only, since POLLIN also takes out-of-band data, which the blocking peek in peer_closed would wait past; an end of stream or a reset is still reported, since WSAPoll sets POLLHUP and POLLERR whatever is asked.
inline bool readable(Socket s) {
    WSAPOLLFD p{};
    p.fd = s;
    p.events = POLLRDNORM;
    return WSAPoll(&p, 1, 0) > 0;
}
// Winsock wants one startup per process; the first listener does it and nothing undoes it, since the process ends with the server.
inline void platform_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) throw std::runtime_error("http: WSAStartup failed");
    });
}
#else
using Socket = int;
constexpr Socket kInvalid = -1;
inline void close_socket(Socket s) { ::close(s); }
inline int last_error() { return errno; }
// The accept failures that belong to one client, a connection aborted before it was taken, an interrupted call or a protocol error on the new connection, after which the next accept can succeed at once.
inline bool per_client(int e) { return e == ECONNABORTED || e == EINTR || e == EPROTO; }
// A failed read that says nothing about the connection: no data yet, or an interrupted call.
inline bool transient(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
// Whether a read would return at once, with data, an end of stream or an error, asked without waiting.
inline bool readable(Socket s) {
    pollfd p{};
    p.fd = s;
    p.events = POLLIN;
    return ::poll(&p, 1, 0) > 0;
}
// A write to a socket the peer has closed raises SIGPIPE and ends the process unless the process ignores it; a client leaving mid-stream is ordinary here, so the signal is ignored once and every send also passes MSG_NOSIGNAL where the platform has it.
inline void platform_init() {
    static std::once_flag once;
    std::call_once(once, [] { signal(SIGPIPE, SIG_IGN); });
}
#endif
#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif
// The peek after a readiness check also passes MSG_DONTWAIT where the platform has it, so a readiness that turns out spurious cannot block the connection thread.
#if defined(MSG_DONTWAIT)
constexpr int kPeekFlags = MSG_PEEK | MSG_DONTWAIT;
#else
constexpr int kPeekFlags = MSG_PEEK;
#endif

struct Request {
    std::string method, path, query;
    std::map<std::string, std::string> headers;   // keys lowercased
    std::string body;
};

// Sizes a request may not exceed; past them the connection answers 413 or 431 and closes rather than reading on.
struct Limits {
    size_t max_header_bytes = 64 * 1024;
    size_t max_body_bytes = 64 * 1024 * 1024;
};

inline const char* reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Unknown";
    }
}

// A client that has left: a write to it failed, or peer_closed found its connection closed.
// Nothing written after that can reach it, so a caller that catches this sends no answer and lets the connection close.
struct ClientGone : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One accepted connection.
// Reads exactly one request, then writes either a whole response or a chunked stream; either way the socket closes with the object, a write to a peer that has gone away throws ClientGone, and peer_closed asks without writing.
class Connection {
public:
    explicit Connection(Socket s) : s_(s) {}
    ~Connection() { close(); }
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&& o) noexcept : s_(o.s_), streaming_(o.streaming_) { o.s_ = kInvalid; }

    // False on a malformed request; `status` then says which answer to send (400, 413, 431).
    // A closed socket before any byte is 0.
    bool read_request(Request& req, const Limits& limits, int& status) {
        status = 0;
        std::string head;
        size_t body_start = std::string::npos;
        while (body_start == std::string::npos) {
            if (head.size() > limits.max_header_bytes) { status = 431; return false; }
            char buf[4096];
            const int n = recv_some(buf, sizeof(buf));
            if (n <= 0) { status = head.empty() ? 0 : 400; return false; }
            head.append(buf, (size_t)n);
            const size_t at = head.find("\r\n\r\n");
            if (at != std::string::npos) body_start = at + 4;
        }
        if (!parse_head(head.substr(0, body_start - 4), req)) { status = 400; return false; }
        size_t length = 0;
        const auto cl = req.headers.find("content-length");
        if (cl != req.headers.end()) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(cl->second.c_str(), &end, 10);
            if (!end || *end != '\0') { status = 400; return false; }
            length = (size_t)v;
        }
        if (req.headers.count("transfer-encoding")) { status = 400; return false; }
        if (length > limits.max_body_bytes) { status = 413; return false; }
        req.body = head.substr(body_start);
        if (req.body.size() > length) { status = 400; return false; }
        while (req.body.size() < length) {
            char buf[16384];
            const size_t want = std::min(sizeof(buf), length - req.body.size());
            const int n = recv_some(buf, (int)want);
            if (n <= 0) { status = 400; return false; }
            req.body.append(buf, (size_t)n);
        }
        return true;
    }

    // Refused while a stream is open: its head has gone out, and a second one would land inside the chunked body.
    void respond(int status, const std::string& content_type, const std::string& body) {
        if (streaming_) throw std::logic_error("http: a response inside a stream");
        std::string out = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
        out += "Content-Type: " + content_type + "\r\n";
        out += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        out += "Connection: close\r\n\r\n";
        out += body;
        send_all(out);
    }

    // A chunked response: the head now, chunks as they come, end_stream() last.
    // Text arrives at the client exactly as written, so a caller that streams UTF-8 must not split a character across chunks.
    void begin_stream(int status, const std::string& content_type) {
        std::string out = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
        out += "Content-Type: " + content_type + "\r\n";
        out += "Transfer-Encoding: chunked\r\n";
        out += "Cache-Control: no-cache\r\n";
        out += "Connection: close\r\n\r\n";
        send_all(out);
        streaming_ = true;
    }
    void write_chunk(const std::string& data) {
        if (data.empty()) return;
        char size[32];
        std::snprintf(size, sizeof(size), "%zx\r\n", data.size());
        send_all(std::string(size) + data + "\r\n");
    }
    void end_stream() {
        if (!streaming_) return;
        send_all("0\r\n\r\n");
        streaming_ = false;
    }

    // Whether the client has closed the connection, asked without blocking and without taking a byte: an end of stream or a reset is closed, nothing to read or data waiting is open.
    // A client that shuts only its sending side after the request reads as closed too, since both arrive as the same end of stream.
    bool peer_closed() {
        if (!readable(s_)) return false;
        char b;
        const int n = (int)::recv(s_, &b, 1, kPeekFlags);
        return n == 0 || (n < 0 && !transient(last_error()));
    }

    void close() {
        if (s_ != kInvalid) { close_socket(s_); s_ = kInvalid; }
    }
    bool open() const { return s_ != kInvalid; }

private:
    int recv_some(char* buf, int n) { return (int)::recv(s_, buf, n, 0); }
    void send_all(const std::string& data) {
        size_t off = 0;
        while (off < data.size()) {
            const int n = (int)::send(s_, data.data() + off, (int)(data.size() - off), kSendFlags);
            if (n <= 0) throw ClientGone("http: the client went away");
            off += (size_t)n;
        }
    }
    static bool parse_head(const std::string& head, Request& req) {
        size_t line_end = head.find("\r\n");
        const std::string line = head.substr(0, line_end);
        const size_t sp1 = line.find(' ');
        const size_t sp2 = line.find(' ', sp1 == std::string::npos ? 0 : sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
        req.method = line.substr(0, sp1);
        std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
        const std::string version = line.substr(sp2 + 1);
        if (version != "HTTP/1.1" && version != "HTTP/1.0") return false;
        if (target.empty() || target[0] != '/') return false;
        const size_t q = target.find('?');
        req.path = target.substr(0, q);
        req.query = q == std::string::npos ? "" : target.substr(q + 1);
        req.headers.clear();
        size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
        while (pos < head.size()) {
            size_t next = head.find("\r\n", pos);
            if (next == std::string::npos) next = head.size();
            const std::string h = head.substr(pos, next - pos);
            pos = next + 2;
            if (h.empty()) continue;
            const size_t colon = h.find(':');
            if (colon == std::string::npos || colon == 0) return false;
            std::string key = h.substr(0, colon), value = h.substr(colon + 1);
            for (auto& c : key) c = (char)std::tolower((unsigned char)c);
            const size_t a = value.find_first_not_of(" \t"), b = value.find_last_not_of(" \t");
            value = a == std::string::npos ? "" : value.substr(a, b - a + 1);
            req.headers[key] = value;
        }
        return true;
    }

    Socket s_;
    bool streaming_ = false;
};

// A listening socket.
// Port 0 asks the system for a free one; port() says which, for tests.
class Listener {
public:
    Listener(const std::string& host, uint16_t port) {
        platform_init();
        s_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s_ == kInvalid) throw std::runtime_error("http: socket failed");
        int one = 1;
#if defined(_WIN32)
        setsockopt(s_, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*)&one, sizeof(one));
#else
        setsockopt(s_, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
#endif
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            close_socket(s_);
            throw std::runtime_error("http: bad host address " + host);
        }
        if (::bind(s_, (const sockaddr*)&addr, sizeof(addr)) != 0) {
            close_socket(s_);
            throw std::runtime_error("http: bind to " + host + ":" + std::to_string(port) + " failed");
        }
        if (::listen(s_, 64) != 0) {
            close_socket(s_);
            throw std::runtime_error("http: listen failed");
        }
        sockaddr_in bound{};
#if defined(_WIN32)
        int len = sizeof(bound);
#else
        socklen_t len = sizeof(bound);
#endif
        if (getsockname(s_, (sockaddr*)&bound, &len) == 0) port_ = ntohs(bound.sin_port);
        else port_ = port;
    }
    ~Listener() { close(); }
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;

    uint16_t port() const { return port_; }

    // Blocks for the next client; an invalid connection means the listener was closed from another thread, which is how the server stops.
    // Any other failure is retried while the listener is open: at once when it was one client's, and after 50 ms otherwise, so a shortage of descriptors or buffers can pass and an error that persists cannot spin a core.
    Connection accept() {
        for (;;) {
            const Socket c = ::accept(s_, nullptr, nullptr);
            if (c != kInvalid) {
                int one = 1;
                setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
                return Connection(c);
            }
            const int e = last_error();
            if (closed_.load()) return Connection(kInvalid);
            if (!per_client(e)) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    void close() {
        closed_.store(true);
        if (s_ != kInvalid) {
#if defined(_WIN32)
            shutdown(s_, SD_BOTH);
#else
            shutdown(s_, SHUT_RDWR);
#endif
            close_socket(s_);
            s_ = kInvalid;
        }
    }

private:
    Socket s_ = kInvalid;
    uint16_t port_ = 0;
    std::atomic<bool> closed_{false};   // set before the socket closes, so a failing accept knows the stop from a passing error
};

// The client side, for the tests: one request, the whole response read to the end, chunked bodies decoded.
// Returns the status; `chunks` receives each chunk as it arrived when the reply was chunked, so a test can see where the server split the stream.
inline int fetch(const std::string& host, uint16_t port, const std::string& method, const std::string& path,
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
