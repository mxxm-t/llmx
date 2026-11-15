#pragma once
// The keys a server admits (docs/SERVER.md, Authentication): Bearer tokens from the file `--api-key-file` names, or every client under `--no-auth`.
// A given key is compared by its SHA-256 digest against every stored digest with no early exit, so neither a key's length nor a matching prefix shows in the time.
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/sha.hpp"
#include "server/http.hpp"

namespace server {

// The fewest characters a key may have, so a guessable key is refused rather than taken.
inline constexpr size_t kMinKeyChars = 16;

class Auth {
public:
    // What a request's Authorization header gave: a key the server holds, no bearer token at all, or a token that matches no key.
    enum class Check { admitted, missing, wrong };

    // A server that admits every client (--no-auth).
    static Auth everyone() {
        Auth a;
        a.open_ = true;
        return a;
    }
    // The keys of the file at `path`, one a line, surrounding whitespace dropped, blank lines and lines starting with # skipped; a bad key names its line.
    // On POSIX `warning`, where given, gets a line when other users may read the file, since a key file is a secret.
    static Auth file(const std::string& path, std::string* warning) {
        std::ifstream in(std::filesystem::u8path(path));
        if (!in) throw std::runtime_error("cannot open the key file " + path);
        Auth a;
        std::string line;
        size_t number = 0;
        while (std::getline(in, line)) {
            ++number;
            const size_t begin = line.find_first_not_of(" \t\r"), end = line.find_last_not_of(" \t\r");
            if (begin == std::string::npos || line[begin] == '#') continue;
            const std::string key = line.substr(begin, end - begin + 1);
            check_key(key, path + ":" + std::to_string(number) + ": ");
            a.digests_.push_back(digest(key));
        }
        if (a.digests_.empty()) throw std::runtime_error("the key file " + path + " holds no key");
#if !defined(_WIN32)
        const auto perms = std::filesystem::status(std::filesystem::u8path(path)).permissions();
        if (warning && (perms & (std::filesystem::perms::group_read | std::filesystem::perms::others_read)) != std::filesystem::perms::none)
            *warning = "server: the key file " + path + " is readable by other users";
#else
        (void)warning;
#endif
        return a;
    }

    bool open() const { return open_; }

    // The request's standing: admitted under an open server or with a key the server holds, else missing or wrong by whether a bearer token was given.
    Check check(const http::Request& req) const {
        if (open_) return Check::admitted;
        const auto it = req.headers.find("authorization");
        if (it == req.headers.end()) return Check::missing;
        const std::string& value = it->second;
        if (value.size() < 7 || !scheme_is_bearer(value) || value[6] != ' ') return Check::missing;
        const size_t a = value.find_first_not_of(' ', 7);
        if (a == std::string::npos) return Check::missing;
        return matches(value.substr(a)) ? Check::admitted : Check::wrong;
    }

    // Whether `key` may be a key: at least kMinKeyChars characters, none of them a space, a control character or outside ASCII, so it travels in a header as it is.
    static void check_key(const std::string& key, const std::string& where) {
        if (key.size() < kMinKeyChars) throw std::runtime_error(where + "a key needs at least " + std::to_string(kMinKeyChars) + " characters");
        for (unsigned char c : key)
            if (c <= ' ' || c >= 0x7f) throw std::runtime_error(where + "a key may hold only printable ASCII characters without spaces");
    }

private:
    static std::string digest(const std::string& key) {
        core::Sha sha(true);
        sha.update(key.data(), key.size());
        return sha.hex();
    }
    static bool scheme_is_bearer(const std::string& value) {
        static const char scheme[] = "bearer";
        for (size_t i = 0; i < 6; ++i)
            if (std::tolower((unsigned char)value[i]) != scheme[i]) return false;
        return true;
    }
    // Every digest is compared whole, the differences accumulated, so the time depends on the number of keys alone.
    bool matches(const std::string& token) const {
        const std::string given = digest(token);
        unsigned any_match = 0;
        for (const std::string& stored : digests_) {
            unsigned diff = 0;
            for (size_t i = 0; i < stored.size(); ++i) diff |= (unsigned)(stored[i] ^ given[i]);
            any_match |= (unsigned)((diff - 1) >> 31);   // 1 when diff is 0, since the digests are the same length
        }
        return any_match != 0;
    }

    bool open_ = false;
    std::vector<std::string> digests_;   // SHA-256 hex digests of the keys, 64 characters each
};

} // namespace server
