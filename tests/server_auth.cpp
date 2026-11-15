// server::Auth (src/server/auth.hpp): the key rule, a key file's lines, the bearer header's reading, every key of a set admitted and any other token refused, and the 401 bodies.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "server/api.hpp"
#include "server/auth.hpp"

namespace {

size_t checks = 0;
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

bool key_refused(const std::string& key, const std::string& text) {
    try { server::Auth::check_key(key, ""); } catch (const std::runtime_error& e) { return std::string(e.what()).find(text) != std::string::npos; }
    return false;
}

std::string refusal(const std::filesystem::path& file) {
    try { (void)server::Auth::file(file.u8string(), nullptr); } catch (const std::runtime_error& e) { return e.what(); }
    return "";
}

void write(const std::filesystem::path& file, const std::string& text) {
    std::ofstream out(file, std::ios::binary);
    out << text;
}

http::Request with(const std::string& authorization, const std::string& path = "/v1/generate") {
    http::Request r;
    r.method = "POST";
    r.path = path;
    if (!authorization.empty()) r.headers["authorization"] = authorization;
    return r;
}

} // namespace

int main() {
    try {
        using Check = server::Auth::Check;
        const std::string first = "first-key-0123456789", second = "second-key-abcdefghij";
        require(key_refused("short", "at least 16 characters"), "a short key refused");
        require(key_refused("a key with spaces in it", "printable ASCII"), "a key with a space refused");
        require(key_refused("tab\tseparated-key-01234", "printable ASCII"), "a key with a control character refused");
        require(key_refused("caf\xc3\xa9-key-0123456789abc", "printable ASCII"), "a key beyond ASCII refused");
        server::Auth::check_key(first, "");
        ++checks;

        // A key file: comments, blank lines, surrounding spaces and CR line ends are not keys.
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("llmx-server-auth-" + std::to_string((unsigned long long)std::hash<std::string>{}(first)));
        std::filesystem::create_directories(dir);
        const std::filesystem::path file = dir / "keys.txt";
        write(file, "# the keys\r\n\r\n  " + first + "  \r\n" + second + "\n");
        std::string warning;
        const server::Auth auth = server::Auth::file(file.u8string(), &warning);
        require(!auth.open(), "a key file does not open the server");
        require(auth.check(with("")) == Check::missing, "no header is missing");
        require(auth.check(with("Basic Zm9vOmJhcg==")) == Check::missing, "another scheme is missing");
        require(auth.check(with("Bearer")) == Check::missing, "the scheme alone is missing");
        require(auth.check(with("Bearer ")) == Check::missing, "the scheme and a space are missing");
        require(auth.check(with("Bearer " + first)) == Check::admitted, "the first key admitted");
        require(auth.check(with("Bearer " + second)) == Check::admitted, "the second key admitted");
        require(auth.check(with("bearer  " + second)) == Check::admitted, "the scheme in lower case and two spaces admitted");
        require(auth.check(with("Bearer " + first + "x")) == Check::wrong, "a key with a character more is wrong");
        require(auth.check(with("Bearer " + first.substr(0, first.size() - 1))) == Check::wrong, "a key with a character less is wrong");
        require(auth.check(with("Bearer " + std::string(first.size(), 'z'))) == Check::wrong, "a key of the same length is wrong");
        require(auth.check(with("Bearer " + first + "  ")) == Check::wrong, "the file's spaces are not part of the key");

        const server::Auth open = server::Auth::everyone();
        require(open.open() && open.check(with("")) == Check::admitted && open.check(with("Bearer nonsense")) == Check::admitted, "an open server admits everything");

        write(file, first + "\nshort\n");
        require(refusal(file).find(":2: a key needs at least 16 characters") != std::string::npos, "a bad key names its line");
        write(file, "# nothing\n\n");
        require(refusal(file).find("holds no key") != std::string::npos, "a file without a key refused");
        require(refusal(dir / "none.txt").find("cannot open") != std::string::npos, "a missing file refused");
        std::filesystem::remove_all(dir);

        // The 401 bodies: the native shape and the compatible routes' shape with the OpenAI clients' type and code.
        const std::string native = server::unauthorized_json(false), compat = server::unauthorized_json(true);
        require(native.rfind("{\"error\":\"", 0) == 0 && native.find("Bearer") != std::string::npos, "the native 401 body");
        require(compat.find("\"type\":\"authentication_error\"") != std::string::npos && compat.find("\"code\":\"invalid_api_key\"") != std::string::npos, "the compatible 401 body");

        std::cout << "server-auth: " << checks << " checks: the key rule, a key file, bearer headers, keys admitted and refused, 401 bodies\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-auth: " << e.what() << "\n";
        return 1;
    }
}
