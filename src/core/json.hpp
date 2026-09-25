#pragma once
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <sstream>
#include <locale>
#include <cmath>
#include <limits>

#include "core/utf8.hpp"

// Minimal recursive-descent JSON parser, written from scratch (no libs).

namespace jmini {

struct Value;
using Object = std::vector<std::pair<std::string, Value>>;
using Array  = std::vector<Value>;

struct Value {
    enum class T { Null, Bool, Number, String, Array, Object } t = T::Null;
    bool b = false;
    double num = 0;
    std::string str;
    Array arr;
    Object obj;

    bool isObject() const { return t == T::Object; }
    bool isArray()  const { return t == T::Array; }
    bool isString() const { return t == T::String; }
    bool isNumber() const { return t == T::Number; }

    const Value* get(const std::string& k) const {
        if (t != T::Object) return nullptr;
        for (const auto& p : obj) if (p.first == k) return &p.second;
        return nullptr;
    }
    const std::string& asString() const { return str; }
    double asNumber() const { return num; }
    const Array& asArray() const { return arr; }
};

class Parser {
    const std::string& s;
    size_t i = 0;

    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) i++; }
    char peek() {
        if (i >= s.size()) throw std::runtime_error("json: unexpected end of input");
        return s[i];
    }
    void expect(char c) { ws(); if (peek() != c) throw std::runtime_error("json: expected character"); i++; }

    uint32_t hex4() {
        if (s.size() - i < 4) throw std::runtime_error("json: truncated Unicode escape");
        uint32_t value = 0;
        for (size_t n = 0; n < 4; ++n) {
            const char c = s[i++];
            unsigned digit;
            if (c >= '0' && c <= '9') digit = unsigned(c - '0');
            else if (c >= 'a' && c <= 'f') digit = unsigned(c - 'a') + 10;
            else if (c >= 'A' && c <= 'F') digit = unsigned(c - 'A') + 10;
            else throw std::runtime_error("json: invalid Unicode escape");
            value = (value << 4) | digit;
        }
        return value;
    }

    void rawUtf8(std::string& out) {
        const size_t start = i - 1;
        const size_t length = utf8::valid_length(s, start);
        if (length == 0) throw std::runtime_error("json: invalid UTF-8");
        out.append(s, start, length);
        i = start + length;
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (i < s.size()) {
            const auto c = static_cast<unsigned char>(s[i++]);
            if (c == '"') return out;
            if (c == '\\') {
                const char e = peek();
                ++i;
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        uint32_t code = hex4();
                        if (code >= 0xd800 && code <= 0xdbff) {
                            if (s.size() - i < 2 || s[i] != '\\' || s[i + 1] != 'u')
                                throw std::runtime_error("json: missing low surrogate");
                            i += 2;
                            const uint32_t low = hex4();
                            if (low < 0xdc00 || low > 0xdfff)
                                throw std::runtime_error("json: invalid low surrogate");
                            code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                        } else if (code >= 0xdc00 && code <= 0xdfff)
                            throw std::runtime_error("json: unpaired low surrogate");
                        out += utf8::encode(code);
                        break;
                    }
                    default: throw std::runtime_error("json: invalid string escape");
                }
            } else if (c < 0x20) throw std::runtime_error("json: unescaped control character");
            else if (c >= 0x80) rawUtf8(out);
            else out += char(c);
        }
        throw std::runtime_error("json: unterminated string");
    }

    Value parseObject(size_t depth) {
        Value v; v.t = Value::T::Object;
        expect('{'); ws();
        if (peek() == '}') { i++; return v; }
        for (;;) {
            ws();
            std::string key = parseString();
            ws(); expect(':');
            Value val = parseValue(depth);
            v.obj.push_back({ std::move(key), std::move(val) });
            ws();
            char c = peek();
            if (c == ',') { i++; continue; }
            if (c == '}') { i++; break; }
            throw std::runtime_error("json: object syntax error");
        }
        return v;
    }

    Value parseArray(size_t depth) {
        Value v; v.t = Value::T::Array;
        expect('['); ws();
        if (peek() == ']') { i++; return v; }
        for (;;) {
            Value val = parseValue(depth);
            v.arr.push_back(std::move(val));
            ws();
            char c = peek();
            if (c == ',') { i++; continue; }
            if (c == ']') { i++; break; }
            throw std::runtime_error("json: array syntax error");
        }
        return v;
    }

    bool digit() const { return i < s.size() && s[i] >= '0' && s[i] <= '9'; }

    void literal(const char* text, size_t length) {
        if (s.compare(i, length, text) != 0) throw std::runtime_error("json: invalid literal");
        i += length;
    }

    Value parseNumber() {
        const size_t start = i;
        if (s[i] == '-') ++i;
        if (!digit()) throw std::runtime_error("json: expected number");
        if (s[i] == '0') ++i;
        else while (digit()) ++i;
        if (i < s.size() && s[i] == '.') {
            ++i;
            if (!digit()) throw std::runtime_error("json: missing fractional digits");
            while (digit()) ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            if (!digit()) throw std::runtime_error("json: missing exponent digits");
            while (digit()) ++i;
        }
        Value v; v.t = Value::T::Number;
        std::istringstream input(s.substr(start, i - start));
        input.imbue(std::locale::classic());
        input >> v.num;
        // Some standard libraries set failbit for representable subnormals or values rounded up to minimum normal.
        const bool tiny = v.num != 0 && std::abs(v.num) <= std::numeric_limits<double>::min();
        if (input.bad() || !input.eof() || !std::isfinite(v.num) || (input.fail() && !tiny))
            throw std::runtime_error("json: number outside double range");
        if (v.num == 0) {
            for (size_t n = start; n < i && s[n] != 'e' && s[n] != 'E'; ++n)
                if (s[n] >= '1' && s[n] <= '9')
                    throw std::runtime_error("json: number underflows double range");
        }
        return v;
    }

    Value parseValue(size_t depth = 0) {
        ws();
        const char c = peek();
        if (c == '{' || c == '[') {
            if (depth == 256) throw std::runtime_error("json: nesting limit exceeded");
            return c == '{' ? parseObject(depth + 1) : parseArray(depth + 1);
        }
        if (c == '"') { Value v; v.t = Value::T::String; v.str = parseString(); return v; }
        if (c == 't') { literal("true", 4); Value v; v.t = Value::T::Bool; v.b = true; return v; }
        if (c == 'f') { literal("false", 5); Value v; v.t = Value::T::Bool; return v; }
        if (c == 'n') { literal("null", 4); return Value(); }
        return parseNumber();
    }

public:
    explicit Parser(const std::string& str) : s(str) {}

    Value parse() {
        ws();
        Value v = parseValue();
        ws();
        if (i != s.size()) throw std::runtime_error("json: trailing data after document");
        return v;
    }
};

inline Value parse(const std::string& s) { return Parser(s).parse(); }

inline std::string quote(const std::string& text) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c < 0x20) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 0x0f];
        } else out += char(c);
    }
    out += '"';
    return out;
}

} // namespace jmini
