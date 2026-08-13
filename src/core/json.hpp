#pragma once
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <cctype>

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

    std::string parseString() {
        expect('"');
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return out;
            if (c == '\\') {
                char e = s[i++];
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
                        if (i + 4 <= s.size()) {
                            unsigned v = (unsigned)std::strtoul(s.substr(i, 4).c_str(), nullptr, 16);
                            out += (char)v;
                            i += 4;
                        }
                        break;
                    }
                    default: out += e;
                }
            } else out += c;
        }
        throw std::runtime_error("json: unterminated string");
    }

    Value parseObject() {
        Value v; v.t = Value::T::Object;
        expect('{'); ws();
        if (peek() == '}') { i++; return v; }
        for (;;) {
            ws();
            std::string key = parseString();
            ws(); expect(':');
            Value val = parseValue();
            v.obj.push_back({ std::move(key), std::move(val) });
            ws();
            char c = peek();
            if (c == ',') { i++; continue; }
            if (c == '}') { i++; break; }
            throw std::runtime_error("json: object syntax error");
        }
        return v;
    }

    Value parseArray() {
        Value v; v.t = Value::T::Array;
        expect('['); ws();
        if (peek() == ']') { i++; return v; }
        for (;;) {
            Value val = parseValue();
            v.arr.push_back(std::move(val));
            ws();
            char c = peek();
            if (c == ',') { i++; continue; }
            if (c == ']') { i++; break; }
            throw std::runtime_error("json: array syntax error");
        }
        return v;
    }

    Value parseValue() {
        ws();
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') { Value v; v.t = Value::T::String; v.str = parseString(); return v; }
        if (c == 't') { expect('t'); expect('r'); expect('u'); expect('e'); Value v; v.t = Value::T::Bool; v.b = true; return v; }
        if (c == 'f') { expect('f'); expect('a'); expect('l'); expect('s'); expect('e'); Value v; v.t = Value::T::Bool; v.b = false; return v; }
        if (c == 'n') { expect('n'); expect('u'); expect('l'); expect('l'); return Value(); }
        std::string num;
        if (c == '-') { num += c; i++; }
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i]=='.'||s[i]=='e'||s[i]=='E'||s[i]=='+'||s[i]=='-')) num += s[i++];
        Value v; v.t = Value::T::Number; v.num = std::strtod(num.c_str(), nullptr); return v;
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

} // namespace jmini
