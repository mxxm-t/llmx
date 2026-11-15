#pragma once
// HTTP/1.1, enough for the server in docs/SERVER.md: listen, accept, read one request with a Content-Length body, write one response or a chunked stream, and tell whether the client has left.
// One thread per connection over a non-blocking socket, held within the deadlines of Limits, no keep-alive beyond one request, no TLS and no external library: a reverse proxy does the rest when the server faces a network.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <list>
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
#include <fcntl.h>
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
constexpr int kShutdownSend = SD_SEND;
// The accept failures that belong to one client, a connection reset before it was taken or an interrupted call, after which the next accept can succeed at once.
inline bool per_client(int e) { return e == WSAECONNRESET || e == WSAEINTR; }
// A failed read that says nothing about the connection: no data yet, or an interrupted call.
inline bool transient(int e) { return e == WSAEWOULDBLOCK || e == WSAEINTR; }
// Waits up to `wait_ms` for a read to return at once, with data, an end of stream or an error, or on a listening socket for a client to take: 1 when one would, 0 when the time ran out, -1 on a failure.
// Normal data only, since POLLIN also takes out-of-band data, which the blocking peek in peer_closed would wait past; an end of stream or a reset is still reported, since WSAPoll sets POLLHUP and POLLERR whatever is asked.
inline int wait_readable(Socket s, int wait_ms) {
    WSAPOLLFD p{};
    p.fd = s;
    p.events = POLLRDNORM;
    return WSAPoll(&p, 1, wait_ms);
}
// Waits up to `wait_ms` for a write to take some bytes at once, or to fail: 1, 0 when the time ran out, -1 on a failure.
inline int wait_writable(Socket s, int wait_ms) {
    WSAPOLLFD p{};
    p.fd = s;
    p.events = POLLWRNORM;
    return WSAPoll(&p, 1, wait_ms);
}
inline bool set_blocking(Socket s, bool blocking) {
    u_long nonblocking = blocking ? 0 : 1;
    return ioctlsocket(s, FIONBIO, &nonblocking) == 0;
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
constexpr int kShutdownSend = SHUT_WR;
// The accept failures that belong to one client, a connection aborted before it was taken, an interrupted call or a protocol error on the new connection, after which the next accept can succeed at once.
inline bool per_client(int e) { return e == ECONNABORTED || e == EINTR || e == EPROTO; }
// A failed read that says nothing about the connection: no data yet, or an interrupted call.
inline bool transient(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINTR; }
// Waits up to `wait_ms` for a read to return at once, with data, an end of stream or an error, or on a listening socket for a client to take: 1 when one would, 0 when the time ran out, -1 on a failure.
inline int wait_readable(Socket s, int wait_ms) {
    pollfd p{};
    p.fd = s;
    p.events = POLLIN;
    return ::poll(&p, 1, wait_ms);
}
// Waits up to `wait_ms` for a write to take some bytes at once, or to fail: 1, 0 when the time ran out, -1 on a failure.
inline int wait_writable(Socket s, int wait_ms) {
    pollfd p{};
    p.fd = s;
    p.events = POLLOUT;
    return ::poll(&p, 1, wait_ms);
}
inline bool set_blocking(Socket s, bool blocking) {
    const int flags = fcntl(s, F_GETFL, 0);
    return flags >= 0 && fcntl(s, F_SETFL, blocking ? flags & ~O_NONBLOCK : flags | O_NONBLOCK) == 0;
}
// A write to a socket the peer has closed raises SIGPIPE, so the signal is ignored once and every send also passes MSG_NOSIGNAL where the platform has it; a client leaving mid-stream is ordinary here.
inline void platform_init() {
    static std::once_flag once;
    std::call_once(once, [] { signal(SIGPIPE, SIG_IGN); });
}
#endif
// Whether a read would return at once, asked without waiting.
inline bool readable(Socket s) { return wait_readable(s, 0) > 0; }
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

// Sizes a request may not exceed, past which the connection answers 413 or 431 and closes rather than reading on, and the deadlines its thread is held to.
// The head has `head_ms` from the first read, the body `body_ms` and `body_ms_per_mib` for each MiB it announces, a quarter of a MiB a second, and a write that no read takes within `send_ms` is a client gone.
struct Limits {
    size_t max_header_bytes = 64 * 1024;
    size_t max_body_bytes = 64 * 1024 * 1024;
    std::chrono::milliseconds head_ms{15000};
    std::chrono::milliseconds body_ms{30000};
    std::chrono::milliseconds body_ms_per_mib{4000};
    std::chrono::milliseconds send_ms{30000};
    std::chrono::milliseconds linger_ms{2000};   // the most a drain after a refusal waits for the client's end of stream (Connection::linger)
};

inline const char* reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
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
    Connection(Connection&& o) noexcept : s_(o.s_), streaming_(o.streaming_), send_ms_(o.send_ms_), linger_ms_(o.linger_ms_), body_length_(o.body_length_) { o.s_ = kInvalid; }

    // The request line and the headers, and of the body what came with them in `req.body`; false on a malformed head, `status` then saying which answer to send (400, 408, 413, 431).
    // A closed socket before any byte is 0, and body_length is what read_body will complete.
    bool read_head(Request& req, const Limits& limits, int& status) {
        status = 0;
        send_ms_ = limits.send_ms;
        linger_ms_ = limits.linger_ms;
        std::string head;
        size_t body_start = std::string::npos;
        auto deadline = std::chrono::steady_clock::now() + limits.head_ms;
        while (body_start == std::string::npos) {
            if (head.size() > limits.max_header_bytes) { status = 431; return false; }
            char buf[4096];
            const int n = recv_some(buf, sizeof(buf), deadline);
            if (n == kTimedOut) { status = 408; return false; }
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
        body_length_ = length;
        return true;
    }
    // The rest of the body read_head announced, within the body deadline; false with `status` 408 when it does not arrive in time and 400 when the client leaves first.
    bool read_body(Request& req, const Limits& limits, int& status) {
        status = 0;
        const auto deadline = std::chrono::steady_clock::now() + limits.body_ms + limits.body_ms_per_mib * (long long)((body_length_ + (1 << 20) - 1) >> 20);
        while (req.body.size() < body_length_) {
            char buf[16384];
            const size_t want = std::min(sizeof(buf), body_length_ - req.body.size());
            const int n = recv_some(buf, (int)want, deadline);
            if (n == kTimedOut) { status = 408; return false; }
            if (n <= 0) { status = 400; return false; }
            req.body.append(buf, (size_t)n);
        }
        return true;
    }

    // Refused while a stream is open: its head has gone out, and a second one would land inside the chunked body.
    // `headers` are further header lines, each ending in CRLF, such as a 401's WWW-Authenticate.
    void respond(int status, const std::string& content_type, const std::string& body, const std::string& headers = "") {
        if (streaming_) throw std::logic_error("http: a response inside a stream");
        std::string out = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
        out += "Content-Type: " + content_type + "\r\n";
        out += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        out += headers;
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

    // For Lingering, which drains a refused connection on its own thread: the sending side shut, then one read without waiting a step, true while the client may still send.
    void shut_send() { ::shutdown(s_, kShutdownSend); }
    bool drain_step() {
        char buf[16384];
        const int n = (int)::recv(s_, buf, sizeof(buf), 0);
        return n > 0 || (n < 0 && transient(last_error()));
    }

    // The body length the head announced, for a reader that decides before read_body.
    size_t body_length() const { return body_length_; }

    // After a response to a request not read in full: the sending side is shut, then what the client still sends is read and dropped until its end of stream or linger_ms_, so the close sends no reset, which would take the response with it.
    // A thread that cannot wait, the accept thread, hands the connection to Lingering instead.
    void linger() {
        ::shutdown(s_, kShutdownSend);
        const auto deadline = std::chrono::steady_clock::now() + linger_ms_;
        char buf[16384];
        while (recv_some(buf, sizeof(buf), deadline) > 0) {}
    }

    void close() {
        if (s_ != kInvalid) { close_socket(s_); s_ = kInvalid; }
    }
    bool open() const { return s_ != kInvalid; }

private:
    static constexpr int kTimedOut = -2;
    // The milliseconds a poll waits at most in one call, so a far deadline is reached in steps and the wait fits the poll's int.
    static constexpr int kPollStepMs = 60000;
    // The milliseconds left to `deadline`, at least 1, for a poll.
    static int step_ms(std::chrono::steady_clock::time_point deadline) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        return (int)std::min<long long>(std::max<long long>(left, 1), kPollStepMs);
    }
    // A read of up to `n` bytes once one would return at once: what recv gives, or kTimedOut once `deadline` has passed with nothing to read.
    // The socket is non-blocking, so a poll that proved spurious gives a transient failure and the wait goes on.
    int recv_some(char* buf, int n, std::chrono::steady_clock::time_point deadline) {
        for (;;) {
            if (std::chrono::steady_clock::now() >= deadline) return kTimedOut;
            const int ready = wait_readable(s_, step_ms(deadline));
            if (ready == 0 || (ready < 0 && transient(last_error()))) continue;
            const int got = (int)::recv(s_, buf, n, 0);
            if (got < 0 && transient(last_error())) continue;
            return got;
        }
    }
    // Writes all of `data` within one send_ms_, each send taking what the socket has room for; a client whose reads do not take it all in time, or that has gone, throws ClientGone.
    void send_all(const std::string& data) {
        size_t off = 0;
        const auto deadline = std::chrono::steady_clock::now() + send_ms_;
        while (off < data.size()) {
            for (;;) {
                const int ready = wait_writable(s_, step_ms(deadline));
                if (ready > 0 || (ready < 0 && !transient(last_error()))) break;
                if (std::chrono::steady_clock::now() >= deadline) throw ClientGone("http: the client read nothing within the send deadline");
            }
            const int n = (int)::send(s_, data.data() + off, (int)(data.size() - off), kSendFlags);
            if (n < 0 && transient(last_error())) continue;
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
    std::chrono::milliseconds send_ms_ = Limits{}.send_ms;   // the send deadline of the limits the request was read with
    std::chrono::milliseconds linger_ms_ = Limits{}.linger_ms;   // the drain deadline of the same limits
    size_t body_length_ = 0;   // the body the head announced
};

// The connections refused by a thread that cannot wait, the accept thread: one thread drains each without waiting and closes it at its end of stream or its deadline, so the close resets nothing.
// The thread starts with the object, since add runs where a thread could not be made, and at most `limit` are held, the oldest closed at once for one more.
class Lingering {
public:
    explicit Lingering(size_t limit) : limit_(limit), t_([this] { run(); }) {}
    ~Lingering() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
        }
        cv_.notify_all();
        t_.join();
    }
    void add(Connection&& c, std::chrono::milliseconds linger) {
        c.shut_send();
        std::lock_guard<std::mutex> g(m_);
        while (items_.size() >= limit_) items_.pop_front();
        items_.push_back({ std::move(c), std::chrono::steady_clock::now() + linger });
        cv_.notify_all();
    }

private:
    struct Item { Connection c; std::chrono::steady_clock::time_point deadline; };
    void run() {
        std::unique_lock<std::mutex> lock(m_);
        while (!stop_) {
            cv_.wait(lock, [&] { return stop_ || !items_.empty(); });
            const auto now = std::chrono::steady_clock::now();
            for (auto it = items_.begin(); it != items_.end();) {
                if (now < it->deadline && it->c.drain_step()) ++it;
                else it = items_.erase(it);
            }
            if (!items_.empty()) cv_.wait_for(lock, std::chrono::milliseconds(20), [&] { return stop_; });
        }
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::list<Item> items_;
    bool stop_ = false;
    const size_t limit_;   // the most held at once
    std::thread t_;        // last, so run() finds the members built
};

// A listening socket; port 0 asks the system for a free one, and port() says which, for tests.
// accept() and close() may run on different threads: accept() holds m_ while it uses the non-blocking socket, which it polls for close() every kCloseCheckMs, and close() takes m_ so it closes the socket only once no accept() uses it.
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
        if (::listen(s_, 64) != 0 || !set_blocking(s_, false)) {
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

    // Blocks for the next client; an invalid connection means the listener was closed, from this thread or another, which is how the server stops.
    // Any other failure is retried while the listener is open, at once when it was one client's or only said none was waiting and after 50 ms otherwise, so a shortage of descriptors can pass and a persistent error cannot spin a core.
    Connection accept() {
        std::lock_guard<std::mutex> hold(m_);
        while (!closed_.load()) {
            const int ready = wait_readable(s_, kCloseCheckMs);
            if (ready == 0) continue;
            if (ready > 0) {
                const Socket c = ::accept(s_, nullptr, nullptr);
                if (c != kInvalid) {
                    // The connection reads and writes after a poll, so its socket is non-blocking and a call never waits past a deadline; Linux does not pass the listener's mode on, so it is set everywhere.
                    if (!set_blocking(c, false)) {
                        close_socket(c);
                        continue;
                    }
                    int one = 1;
                    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
                    return Connection(c);
                }
            }
            const int e = last_error();
            if (!per_client(e) && !transient(e)) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return Connection(kInvalid);
    }
    // Ends every accept() within kCloseCheckMs and leaves the socket to close(); it only stores a flag, so a signal handler may call it.
    void request_close() { closed_.store(true); }
    // Ends every accept(), then closes the socket once none uses it, which takes up to kCloseCheckMs while one waits for a client.
    void close() {
        closed_.store(true);
        std::lock_guard<std::mutex> hold(m_);
        if (s_ != kInvalid) {
            close_socket(s_);
            s_ = kInvalid;
        }
    }

private:
    static constexpr int kCloseCheckMs = 100;
    Socket s_ = kInvalid;
    uint16_t port_ = 0;
    std::atomic<bool> closed_{false};   // set by close() before it takes m_, which a waiting accept() holds until it sees this
    std::mutex m_;   // held by accept() while it uses the socket and by close() to close it
};

} // namespace http
