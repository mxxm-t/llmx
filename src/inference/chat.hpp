#pragma once
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/utf8.hpp"
#include "format/gguf.hpp"
#include "tokenizer/tokenizer.hpp"

// Chat templates, a GGUF file's `tokenizer.chat_template` in the Jinja language, rendered byte for byte as the reference renderer of docs/src/inference-chat.md renders them.
// A template outside the part of Jinja that page lists is refused when it is parsed, and only `chat` and `serve` raise the refusal.

namespace chat {

// A template the renderer does not take: a syntax error, or a tag, filter, test, method or global outside the part of Jinja it supports.
struct Refused : std::runtime_error { using std::runtime_error::runtime_error; };
// A render that fails where the reference renderer fails: the template's raise_exception with the template's own message, an undefined value used, or an operation Python refuses.
// It is also raised where this renderer cannot follow Python, such as a number past 64 bits or case mapping beyond ASCII, rather than render other text.
struct TemplateError : std::runtime_error { using std::runtime_error::runtime_error; };

// One turn of a conversation.
// `reasoning_content` is an assistant turn's reasoning, held apart from its reply as clients pass it, and absent when the turn has none.
struct Message {
    std::string role;
    std::string content;
    std::optional<std::string> reasoning_content;
};

// An assistant turn split as the Qwen templates split one themselves: the reply after the last </think> without the newlines that open it, and the reasoning before the first </think>, after the last <think> there, without the newlines around it.
// A text without </think> is all reply and has no reasoning.
// This is the one owner of the split, which ChatFormat::assistant applies, where the template needs it, for chat's replies and the server's assistant messages alike.
inline Message assistant_turn(const std::string& text) {
    Message m{ "assistant", text, std::nullopt };
    const size_t first = text.find("</think>");
    if (first == std::string::npos) return m;
    std::string reasoning = text.substr(0, first);
    reasoning.erase(reasoning.find_last_not_of('\n') + 1);
    const size_t open = reasoning.rfind("<think>");
    if (open != std::string::npos) reasoning.erase(0, open + 7);
    reasoning.erase(0, std::min(reasoning.find_first_not_of('\n'), reasoning.size()));
    std::string reply = text.substr(text.rfind("</think>") + 8);
    reply.erase(0, std::min(reply.find_first_not_of('\n'), reply.size()));
    m.content = std::move(reply);
    m.reasoning_content = std::move(reasoning);
    return m;
}

namespace jj {

// The longest string or list one operation builds, and the most loop iterations and macro calls one render runs: far past any chat prompt, so a template that would run away fails instead.
constexpr size_t kMaxBuilt = size_t(1) << 28;
constexpr uint64_t kMaxSteps = 10000000;
// How deep values may nest, containers in containers, which printing, comparing and freeing one follow recursively on the host stack.
constexpr int kMaxValueDepth = 100;
// How deeply a template's statements and expressions may nest where it is written, and how deeply a render may nest them, macro calls included.
// At these limits a parse takes at most about 100 KB of host stack and a render 200 KB, far inside the 512 KiB of the smallest thread stack a render runs on, where the Qwen templates nest about 20 deep.
constexpr int kMaxNest = 100;
constexpr int kMaxRenderDepth = 250;

// Python raises for these, so a render that reaches one fails here too.
[[noreturn]] inline void type_error(const std::string& what) { throw TemplateError(what); }

// ---- text, as Python's str methods see it ----

// The code point at s[i], with its length in bytes in `len`.
// A byte that starts no valid UTF-8 character is a character of its own, U+DC80 to U+DCFF as Python's surrogateescape reads it.
inline uint32_t decode_at(const std::string& s, size_t i, size_t& len) {
    const auto c = static_cast<unsigned char>(s[i]);
    len = utf8::valid_length(s, i);
    if (len == 0) { len = 1; return 0xDC00u | c; }
    if (len == 1) return c;
    uint32_t cp = c & (len == 2 ? 0x1Fu : len == 3 ? 0x0Fu : 0x07u);
    for (size_t n = 1; n < len; ++n) cp = (cp << 6) | (static_cast<unsigned char>(s[i + n]) & 0x3Fu);
    return cp;
}

// Where each character of s starts, then s.size(): character k is the bytes from at[k] to at[k + 1].
inline std::vector<size_t> char_offsets(const std::string& s) {
    std::vector<size_t> at;
    size_t len = 0;
    for (size_t i = 0; i < s.size(); i += len) { at.push_back(i); decode_at(s, i, len); }
    at.push_back(s.size());
    return at;
}

inline size_t char_count(const std::string& s) { return char_offsets(s).size() - 1; }

// Python's str.isspace, which is also the whitespace of a template's own rules.
inline bool is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// Python's str.isprintable over what chat text holds: false for controls, separators other than the space, format characters, surrogates, private use and noncharacters.
// Unassigned code points, which Python also prints escaped, print as they are here.
inline bool is_printable(uint32_t c) {
    if (c < 0x20 || (c >= 0x7F && c <= 0xA0) || c == 0xAD) return false;
    if ((c >= 0x0600 && c <= 0x0605) || c == 0x061C || c == 0x06DD || c == 0x070F || c == 0x08E2 || c == 0x1680 || c == 0x180E) return false;
    if ((c >= 0x2000 && c <= 0x200F) || (c >= 0x2028 && c <= 0x202F) || (c >= 0x205F && c <= 0x2064) ||
        (c >= 0x2066 && c <= 0x206F) || c == 0x3000 || c == 0xFEFF || (c >= 0xFFF9 && c <= 0xFFFB)) return false;
    if ((c >= 0xD800 && c <= 0xF8FF) || c >= 0xF0000 || (c & 0xFFFE) == 0xFFFE || (c >= 0xFDD0 && c <= 0xFDEF)) return false;
    if (c == 0xE0001 || (c >= 0xE0020 && c <= 0xE007F)) return false;
    return true;
}

// str.strip, lstrip and rstrip: `chars` is the set of characters to remove, Python's whitespace when absent.
inline std::string py_strip(const std::string& s, const std::string* chars, bool left, bool right) {
    std::vector<uint32_t> set;
    size_t len = 0;
    if (chars) for (size_t i = 0; i < chars->size(); i += len) set.push_back(decode_at(*chars, i, len));
    const auto at = char_offsets(s);
    const auto removable = [&](size_t k) {
        const uint32_t c = decode_at(s, at[k], len);
        return chars ? std::find(set.begin(), set.end(), c) != set.end() : is_space(c);
    };
    size_t a = 0, b = at.size() - 1;
    if (left) while (a < b && removable(a)) ++a;
    if (right) while (b > a && removable(b - 1)) --b;
    return s.substr(at[a], at[b] - at[a]);
}

// str.split: at every `sep`, or when it is absent at runs of whitespace with no empty parts; at most `limit` splits when it is not negative.
inline std::vector<std::string> py_split(const std::string& s, const std::string* sep, int64_t limit) {
    std::vector<std::string> out;
    if (sep) {
        if (sep->empty()) type_error("empty separator");
        size_t pos = 0;
        while (limit < 0 || int64_t(out.size()) < limit) {
            const size_t hit = s.find(*sep, pos);
            if (hit == std::string::npos) break;
            out.push_back(s.substr(pos, hit - pos));
            pos = hit + sep->size();
        }
        out.push_back(s.substr(pos));
        return out;
    }
    const auto at = char_offsets(s);
    const size_t n = at.size() - 1;
    size_t len = 0;
    const auto space = [&](size_t k) { return is_space(decode_at(s, at[k], len)); };
    for (size_t k = 0;;) {
        while (k < n && space(k)) ++k;
        if (k == n) break;
        if (limit >= 0 && int64_t(out.size()) == limit) { out.push_back(py_strip(s.substr(at[k]), nullptr, false, true)); break; }
        size_t e = k;
        while (e < n && !space(e)) ++e;
        out.push_back(s.substr(at[k], at[e] - at[k]));
        k = e;
    }
    return out;
}

// str.replace, where an empty `old` matches before every character and at the end.
inline std::string py_replace(const std::string& s, const std::string& old, const std::string& with, int64_t count) {
    std::string out;
    int64_t done = 0;
    const auto more = [&] { return count < 0 || done < count; };
    if (old.empty()) {
        const auto at = char_offsets(s);
        for (size_t k = 0; k + 1 < at.size(); ++k) {
            if (more()) { out += with; ++done; }
            out.append(s, at[k], at[k + 1] - at[k]);
            if (out.size() > kMaxBuilt) type_error("a replace built too long a string");
        }
        if (more()) out += with;
        return out;
    }
    size_t pos = 0;
    for (size_t hit; more() && (hit = s.find(old, pos)) != std::string::npos; ++done) {
        out.append(s, pos, hit - pos);
        out += with;
        pos = hit + old.size();
        if (out.size() > kMaxBuilt) type_error("a replace built too long a string");
    }
    return out + s.substr(pos);
}

// Case mapping, which here covers ASCII letters; text holding any other character fails rather than map differently from Python.
inline void ascii_only(const std::string& s, const std::string& what) {
    for (char c : s)
        if (static_cast<unsigned char>(c) >= 0x80) type_error(what + " of text beyond ASCII is not supported");
}
inline bool ascii_letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline std::string ascii_case(std::string s, bool upper, const std::string& what) {
    ascii_only(s, what);
    for (auto& c : s) if (upper ? (c >= 'a' && c <= 'z') : (c >= 'A' && c <= 'Z')) c = char(c ^ 0x20);
    return s;
}
// str.title: a letter after a letter is lowered, any other letter raised.
inline std::string py_title(std::string s) {
    ascii_only(s, "title");
    bool after_letter = false;
    for (auto& c : s) {
        const bool letter = ascii_letter(c);
        if (letter && after_letter == (c >= 'A' && c <= 'Z')) c = char(c ^ 0x20);
        after_letter = letter;
    }
    return s;
}
// Jinja's title filter: each word, split at runs of '-', whitespace, '(', '{', '[' and '<', has its first character raised and the rest lowered.
inline std::string title_filter(const std::string& s) {
    ascii_only(s, "title");
    std::string out;
    const auto separator = [](char c) { return c == '-' || c == '(' || c == '{' || c == '[' || c == '<' || is_space(uint32_t(static_cast<unsigned char>(c))); };
    for (size_t i = 0; i < s.size();) {
        const size_t start = i;
        const bool sep = separator(s[i]);
        while (i < s.size() && separator(s[i]) == sep) ++i;
        std::string part = s.substr(start, i - start);
        if (!sep) {
            part = ascii_case(part, false, "title");
            if (part[0] >= 'a' && part[0] <= 'z') part[0] = char(part[0] ^ 0x20);
        }
        out += part;
    }
    return out;
}

// Python's repr of a string: single quotes unless the text holds one and no double quote, with controls and unprintable characters escaped.
inline std::string str_repr(const std::string& s) {
    const char q = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
    std::string out(1, q);
    char buf[16];
    size_t len = 0;
    for (size_t i = 0; i < s.size(); i += len) {
        const uint32_t c = decode_at(s, i, len);
        if (c == uint32_t(q) || c == '\\') { out += '\\'; out += char(c); }
        else if (c == '\t') out += "\\t";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c >= 0x20 && c < 0x7F) out += char(c);
        else if (is_printable(c)) out.append(s, i, len);
        else {
            std::snprintf(buf, sizeof buf, c <= 0xFF ? "\\x%02x" : c <= 0xFFFF ? "\\u%04x" : "\\U%08x", unsigned(c));
            out += buf;
        }
    }
    return out + q;
}

// Python's repr of a float: the shortest digits that read back to it, fixed from 1e-4 to below 1e16 and in exponent form outside.
inline std::string float_repr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
    char buf[40];
    for (int precision = 1; precision <= 17; ++precision) {
        std::snprintf(buf, sizeof buf, "%.*e", precision - 1, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    // buf now holds [-]d[.ddd]e(+|-)XX.
    std::string text(buf), digits;
    const bool negative = text[0] == '-';
    const size_t e = text.find('e');
    for (size_t k = negative ? 1 : 0; k < e; ++k) if (text[k] != '.') digits += text[k];
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int exponent = std::atoi(text.c_str() + e + 1);
    std::string out = negative ? "-" : "";
    if (exponent < -4 || exponent >= 16) {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        std::snprintf(buf, sizeof buf, "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
        return out + buf;
    }
    if (exponent < 0) return out + "0." + std::string(size_t(-exponent - 1), '0') + digits;
    const size_t whole = size_t(exponent) + 1;
    if (digits.size() <= whole) return out + digits + std::string(whole - digits.size(), '0') + ".0";
    return out + digits.substr(0, whole) + "." + digits.substr(whole);
}

// ---- values ----

struct Value;
struct Scope;
struct MacroNode;

// Names mapped to values in the order they were set, as a Python dict keeps them.
struct Object {
    std::vector<std::pair<std::string, Value>> items;
    const Value* find(const std::string& key) const;
    void set(const std::string& key, Value v);
};

// A macro with the scope it was defined in, which it reads when it is called, as a closure reads its scope.
// The scope is held weakly, since it holds the macro.
struct Closure {
    const MacroNode* node;
    std::weak_ptr<Scope> scope;
};

[[noreturn]] inline void nested_too_deeply() { type_error("a value nested more than " + std::to_string(kMaxValueDepth) + " deep is not supported"); }

struct Value {
    // UNDEF is Jinja's undefined; VIEW is a dict's keys(), values() or items(); LOOP is the loop object, NS a namespace; FUNC is a global function, METHOD a method bound to its object.
    enum Kind : uint8_t { UNDEF, NONE, BOOL, INT, FLOAT, STR, LIST, TUPLE, DICT, VIEW, NS, LOOP, MACRO, FUNC, METHOD };
    Kind k = UNDEF;
    bool b = false;
    // How many containers deep the value nests, and whether a namespace is in it, both fixed when a container is built: a namespace holds no namespace, so no value holds itself.
    uint16_t depth = 0;
    bool holds_ns = false;
    int64_t i = 0;
    double f = 0;
    std::string s;                                  // STR's text; the error an UNDEF raises when it is used; a FUNC's or METHOD's name; a VIEW's class
    std::shared_ptr<const std::vector<Value>> seq;  // LIST, TUPLE and VIEW; a METHOD's object
    std::shared_ptr<Object> obj;                    // DICT, NS and LOOP
    std::shared_ptr<const Closure> macro;           // MACRO

    static Value undefined(std::string message) { Value v; v.s = std::move(message); return v; }
    static Value none() { Value v; v.k = NONE; return v; }
    static Value boolean(bool x) { Value v; v.k = BOOL; v.b = x; return v; }
    static Value integer(int64_t x) { Value v; v.k = INT; v.i = x; return v; }
    static Value number(double x) { Value v; v.k = FLOAT; v.f = x; return v; }
    static Value str(std::string x) { Value v; v.k = STR; v.s = std::move(x); return v; }
    static Value list(std::vector<Value> items, bool tuple = false) { return sequence_of(tuple ? TUPLE : LIST, std::move(items)); }
    // A dict's keys(), values() or items(), named by `cls` as Python names the view's class.
    static Value view(std::string cls, std::vector<Value> items) {
        Value v = sequence_of(VIEW, std::move(items));
        v.s = std::move(cls);
        return v;
    }
    static Value dict(std::shared_ptr<Object> o, Kind kind = DICT);
    static Value func(std::string name) { Value v; v.k = FUNC; v.s = std::move(name); return v; }
    static Value method(std::string name, const Value& self) {
        Value v = sequence_of(METHOD, { self });
        v.s = std::move(name);
        return v;
    }

    bool numeric() const { return k == BOOL || k == INT || k == FLOAT; }
    bool integral() const { return k == BOOL || k == INT; }
    int64_t as_int() const { return k == BOOL ? int64_t(b) : i; }
    double as_float() const { return k == FLOAT ? f : double(as_int()); }
    bool sequence() const { return k == LIST || k == TUPLE; }
    // A view that compares as a set, as keys() and items() do.
    bool set_like() const { return k == VIEW && s != "dict_values"; }

    // Takes in an item of a container being built.
    void hold(const Value& item) {
        if (item.depth + 1 > depth) depth = uint16_t(item.depth + 1);
        holds_ns = holds_ns || item.holds_ns;
        if (depth > kMaxValueDepth) nested_too_deeply();
    }

private:
    static Value sequence_of(Kind kind, std::vector<Value> items) {
        Value v;
        v.k = kind;
        v.depth = 1;
        for (const auto& item : items) v.hold(item);
        v.seq = std::make_shared<const std::vector<Value>>(std::move(items));
        return v;
    }
};

inline const Value* Object::find(const std::string& key) const {
    for (const auto& kv : items) if (kv.first == key) return &kv.second;
    return nullptr;
}
inline void Object::set(const std::string& key, Value v) {
    for (auto& kv : items) if (kv.first == key) { kv.second = std::move(v); return; }
    items.emplace_back(key, std::move(v));
}

inline Value Value::dict(std::shared_ptr<Object> o, Kind kind) {
    Value v;
    v.k = kind;
    v.depth = 1;
    for (const auto& kv : o->items) v.hold(kv.second);
    if (kind == NS && v.holds_ns) type_error("a namespace holding a namespace is not supported");
    v.holds_ns = v.holds_ns || kind == NS;
    v.obj = std::move(o);
    return v;
}

// The Python class of a value, as Python's own messages name it.
inline const char* type_name(const Value& v) {
    switch (v.k) {
        case Value::UNDEF: return "Undefined";
        case Value::NONE: return "NoneType";
        case Value::BOOL: return "bool";
        case Value::INT: return "int";
        case Value::FLOAT: return "float";
        case Value::STR: return "str";
        case Value::LIST: return "list";
        case Value::TUPLE: return "tuple";
        case Value::DICT: return "dict";
        case Value::VIEW: return v.s.c_str();
        case Value::NS: return "Namespace";
        case Value::LOOP: return "LoopContext";
        case Value::MACRO: return "Macro";
        case Value::FUNC: return "function";
        case Value::METHOD: return "builtin_function_or_method";
    }
    return "object";
}

// How Jinja names an object in an undefined value's message.
inline std::string object_type_repr(const Value& v) {
    switch (v.k) {
        case Value::NONE: return "None";
        case Value::NS: return "jinja2.utils.Namespace object";
        case Value::LOOP: return "jinja2.runtime.LoopContext object";
        case Value::MACRO: return "jinja2.runtime.Macro object";
        default: return std::string(type_name(v)) + " object";
    }
}

inline std::string repr(const Value& v);

// Jinja's undefined values, each carrying the message it raises once it is used.
inline Value undefined_name(const std::string& name) { return Value::undefined(str_repr(name) + " is undefined"); }
inline Value undefined_attr(const Value& obj, const std::string& name) {
    return Value::undefined(str_repr(object_type_repr(obj)) + " has no attribute " + str_repr(name));
}
inline Value undefined_item(const Value& obj, const Value& key) {
    if (key.k == Value::STR) return undefined_attr(obj, key.s);
    return Value::undefined(object_type_repr(obj) + " has no element " + repr(key));
}
inline Value unsafe_attr(const Value& obj, const std::string& name) {
    return Value::undefined("access to attribute " + str_repr(name) + " of " + str_repr(type_name(obj)) + " object is unsafe.");
}

[[noreturn]] inline void undefined_error(const Value& v) { throw TemplateError(v.s); }

inline bool truthy(const Value& v) {
    switch (v.k) {
        case Value::UNDEF: case Value::NONE: return false;
        case Value::BOOL: return v.b;
        case Value::INT: return v.i != 0;
        case Value::FLOAT: return v.f != 0;
        case Value::STR: return !v.s.empty();
        case Value::LIST: case Value::TUPLE: case Value::VIEW: return !v.seq->empty();
        case Value::DICT: return !v.obj->items.empty();
        default: return true;
    }
}

// Python's str().
inline std::string to_str(const Value& v) {
    switch (v.k) {
        case Value::UNDEF: return "";
        case Value::NONE: return "None";
        case Value::BOOL: return v.b ? "True" : "False";
        case Value::INT: return std::to_string(v.i);
        case Value::FLOAT: return float_repr(v.f);
        case Value::STR: return v.s;
        default: return repr(v);
    }
}

inline std::string dict_repr(const Object& o) {
    std::string out = "{";
    for (size_t n = 0; n < o.items.size(); ++n) {
        if (n) out += ", ";
        out += str_repr(o.items[n].first) + ": " + repr(o.items[n].second);
    }
    return out + "}";
}

// Python's repr(), which a list or dict prints its items with.
inline std::string repr(const Value& v) {
    switch (v.k) {
        case Value::UNDEF: return "Undefined";
        case Value::STR: return str_repr(v.s);
        case Value::LIST: case Value::TUPLE: case Value::VIEW: {
            std::string out = v.k == Value::TUPLE ? "(" : v.k == Value::VIEW ? v.s + "([" : "[";
            for (size_t n = 0; n < v.seq->size(); ++n) { if (n) out += ", "; out += repr((*v.seq)[n]); }
            if (v.k == Value::TUPLE && v.seq->size() == 1) out += ",";
            return out + (v.k == Value::TUPLE ? ")" : v.k == Value::VIEW ? "])" : "]");
        }
        case Value::DICT: return dict_repr(*v.obj);
        case Value::NS: return "<Namespace " + dict_repr(*v.obj) + ">";
        case Value::LOOP: return "<LoopContext " + to_str(*v.obj->find("index")) + "/" + to_str(*v.obj->find("length")) + ">";
        case Value::MACRO: return "<Macro " + str_repr(v.s) + ">";
        case Value::FUNC: return "<function " + v.s + ">";
        case Value::METHOD: return "<built-in method " + v.s + ">";
        default: return to_str(v);
    }
}

inline bool contains(const Value& container, const Value& item);

// Python's ==; Jinja's undefined equals only another undefined.
// Two views of keys or items are equal as sets are, and a view of values equals only itself.
inline bool equal(const Value& a, const Value& b) {
    if (a.numeric() && b.numeric())
        return a.k == Value::FLOAT || b.k == Value::FLOAT ? a.as_float() == b.as_float() : a.as_int() == b.as_int();
    if (a.k != b.k) return false;
    switch (a.k) {
        case Value::UNDEF: case Value::NONE: return true;
        case Value::STR: return a.s == b.s;
        case Value::LIST: case Value::TUPLE: {
            if (a.seq->size() != b.seq->size()) return false;
            for (size_t n = 0; n < a.seq->size(); ++n) if (!equal((*a.seq)[n], (*b.seq)[n])) return false;
            return true;
        }
        case Value::VIEW: {
            if (!a.set_like() || !b.set_like()) return a.seq == b.seq;
            if (a.seq->size() != b.seq->size()) return false;
            for (const auto& x : *a.seq) if (!contains(b, x)) return false;
            return true;
        }
        case Value::DICT: {
            if (a.obj->items.size() != b.obj->items.size()) return false;
            for (const auto& kv : a.obj->items) {
                const Value* other = b.obj->find(kv.first);
                if (!other || !equal(kv.second, *other)) return false;
            }
            return true;
        }
        case Value::NS: case Value::LOOP: return a.obj == b.obj;
        case Value::MACRO: return a.macro == b.macro;
        case Value::FUNC: return a.s == b.s;
        default: return false;
    }
}

// Python's <, <=, > and >= (`op`), for numbers, strings, sequences, and views of keys or items, which order as sets do, by inclusion; anything else fails as Python fails, naming the operator.
inline bool compare(const std::string& op, const Value& a, const Value& b) {
    const auto order = [&](auto x, auto y) { return op == "<" ? x < y : op == "<=" ? x <= y : op == ">" ? x > y : x >= y; };
    if (a.numeric() && b.numeric())
        return a.k == Value::FLOAT || b.k == Value::FLOAT ? order(a.as_float(), b.as_float()) : order(a.as_int(), b.as_int());
    if (a.k == Value::STR && b.k == Value::STR) return order(a.s.compare(b.s), 0);
    if (a.k == b.k && a.sequence()) {
        const auto& x = *a.seq;
        const auto& y = *b.seq;
        for (size_t n = 0; n < x.size() && n < y.size(); ++n)
            if (!equal(x[n], y[n])) return compare(op, x[n], y[n]);
        return order(x.size(), y.size());
    }
    if (a.set_like() && b.set_like()) {
        const auto within = [](const Value& p, const Value& q) {
            for (const auto& x : *p.seq) if (!contains(q, x)) return false;
            return true;
        };
        const size_t m = a.seq->size(), n = b.seq->size();
        if (op == "<") return m < n && within(a, b);
        if (op == "<=") return m <= n && within(a, b);
        if (op == ">") return m > n && within(b, a);
        return m >= n && within(b, a);
    }
    if (a.k == Value::UNDEF) undefined_error(a);
    if (b.k == Value::UNDEF) undefined_error(b);
    type_error("'" + op + "' not supported between instances of '" + type_name(a) + "' and '" + type_name(b) + "'");
}
inline bool less(const Value& a, const Value& b) { return compare("<", a, b); }

// What `for` walks: a sequence's items, a dict's keys, a string's characters, and nothing for undefined.
inline std::vector<Value> iterate(const Value& v) {
    switch (v.k) {
        case Value::UNDEF: return {};
        case Value::LIST: case Value::TUPLE: case Value::VIEW: return *v.seq;
        case Value::DICT: {
            std::vector<Value> keys;
            for (const auto& kv : v.obj->items) keys.push_back(Value::str(kv.first));
            return keys;
        }
        case Value::STR: {
            const auto at = char_offsets(v.s);
            std::vector<Value> chars;
            for (size_t n = 0; n + 1 < at.size(); ++n) chars.push_back(Value::str(v.s.substr(at[n], at[n + 1] - at[n])));
            return chars;
        }
        default: type_error(std::string("'") + type_name(v) + "' object is not iterable");
    }
}

inline int64_t length(const Value& v) {
    switch (v.k) {
        case Value::UNDEF: return 0;
        case Value::STR: return int64_t(char_count(v.s));
        case Value::LIST: case Value::TUPLE: case Value::VIEW: return int64_t(v.seq->size());
        case Value::DICT: return int64_t(v.obj->items.size());
        case Value::LOOP: return v.obj->find("length")->i;
        default: type_error(std::string("object of type '") + type_name(v) + "' has no len()");
    }
}

// A dict key must hash, as a list, a dict and a view do not.
inline void hashable(const Value& key) {
    if (key.k == Value::LIST || key.k == Value::DICT || key.k == Value::VIEW) type_error(std::string("unhashable type: '") + type_name(key) + "'");
}

// Python's `in`: a substring, an item, a dict key, or in a view of items a (key, value) tuple.
inline bool contains(const Value& container, const Value& item) {
    switch (container.k) {
        case Value::UNDEF: return false;
        case Value::STR:
            if (item.k != Value::STR) type_error(std::string("'in <string>' requires string as left operand, not ") + type_name(item));
            return container.s.find(item.s) != std::string::npos;
        case Value::LIST: case Value::TUPLE:
            for (const auto& x : *container.seq) if (equal(x, item)) return true;
            return false;
        case Value::VIEW:
            if (container.s == "dict_keys") hashable(item);
            if (container.s == "dict_items") {
                if (item.k != Value::TUPLE || item.seq->size() != 2) return false;
                hashable((*item.seq)[0]);
            }
            for (const auto& x : *container.seq) if (equal(x, item)) return true;
            return false;
        case Value::DICT:
            hashable(item);
            return item.k == Value::STR && container.obj->find(item.s);
        default: type_error(std::string("argument of type '") + type_name(container) + "' is not iterable");
    }
}

// Items start, start + step, ... of a string or a sequence of n items, by Python's slice rules.
struct SliceRange { int64_t start, step, count; };
inline SliceRange slice_range(int64_t n, const Value& start, const Value& stop, const Value& step) {
    for (const Value* v : { &start, &stop, &step })
        if (!v->integral() && v->k != Value::NONE) type_error("slice indices must be integers or None or have an __index__ method");
    const int64_t st = step.k == Value::NONE ? 1 : step.as_int();
    if (st == 0) type_error("slice step cannot be zero");
    if (st == std::numeric_limits<int64_t>::min()) type_error("an integer past 64 bits is not supported");
    const auto clamp = [&](const Value& v, int64_t missing) {
        if (v.k == Value::NONE) return missing;
        int64_t x = v.as_int();
        if (x < 0) { x = x < -n ? -1 : x + n; if (x < 0) x = st < 0 ? -1 : 0; }
        else if (x >= n) x = st < 0 ? n - 1 : n;
        return x;
    };
    const int64_t a = clamp(start, st < 0 ? n - 1 : 0), b = clamp(stop, st < 0 ? -1 : n);
    int64_t count = 0;
    if (st > 0 && a < b) count = (b - a - 1) / st + 1;
    if (st < 0 && b < a) count = (a - b - 1) / -st + 1;
    return { a, st, count };
}

inline bool one_of(const std::string& s, std::initializer_list<const char*> names) {
    for (const char* n : names) if (s == n) return true;
    return false;
}

// The attributes that are methods of a str.
inline bool str_attribute(const std::string& n) {
    return one_of(n, { "capitalize", "casefold", "center", "count", "encode", "endswith", "expandtabs", "find", "format",
                       "format_map", "index", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier",
                       "islower", "isnumeric", "isprintable", "isspace", "istitle", "isupper", "join", "ljust", "lower",
                       "lstrip", "maketrans", "partition", "removeprefix", "removesuffix", "replace", "rfind", "rindex",
                       "rjust", "rpartition", "rsplit", "rstrip", "split", "splitlines", "startswith", "strip", "swapcase",
                       "title", "translate", "upper", "zfill" });
}

// Jinja's attribute access in the sandbox: the object's own attribute (a method) first, then its item, else undefined.
// The methods that change a dict or a list are unsafe there and read as undefined.
inline Value get_attr(const Value& o, const std::string& name) {
    switch (o.k) {
        case Value::UNDEF: undefined_error(o);
        case Value::DICT:
            if (one_of(name, { "copy", "fromkeys", "get", "items", "keys", "values" })) return Value::method(name, o);
            if (one_of(name, { "clear", "pop", "popitem", "setdefault", "update" })) return unsafe_attr(o, name);
            if (const Value* v = o.obj->find(name)) return *v;
            return undefined_attr(o, name);
        case Value::NS: case Value::LOOP:
            if (const Value* v = o.obj->find(name)) return *v;
            return undefined_attr(o, name);
        case Value::STR: return str_attribute(name) ? Value::method(name, o) : undefined_attr(o, name);
        case Value::LIST: case Value::TUPLE:
            if (one_of(name, { "count", "index" }) || (o.k == Value::LIST && name == "copy")) return Value::method(name, o);
            if (o.k == Value::LIST && one_of(name, { "append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort" })) return unsafe_attr(o, name);
            return undefined_attr(o, name);
        default: return undefined_attr(o, name);
    }
}

// Jinja's subscript in the sandbox: the item first, then for a string key the attribute, else undefined.
inline Value get_item(const Value& o, const Value& key) {
    if (o.k == Value::UNDEF) undefined_error(o);
    if (key.integral() && (o.sequence() || o.k == Value::STR)) {
        const auto at = o.k == Value::STR ? char_offsets(o.s) : std::vector<size_t>();
        const int64_t n = o.k == Value::STR ? int64_t(at.size()) - 1 : int64_t(o.seq->size());
        int64_t x = key.as_int();
        if (x < 0 && x >= -n) x += n;
        if (x < 0 || x >= n) return undefined_item(o, key);
        if (o.k == Value::STR) return Value::str(o.s.substr(at[size_t(x)], at[size_t(x) + 1] - at[size_t(x)]));
        return (*o.seq)[size_t(x)];
    }
    if (key.k != Value::STR) return undefined_item(o, key);
    if (o.k == Value::DICT) if (const Value* v = o.obj->find(key.s)) return *v;
    return get_attr(o, key.s);
}

// ---- JSON, as the reference renderer's tojson writes it with Python's json.dumps ----

struct JsonStyle {
    bool ensure_ascii = false;
    bool indented = false;
    std::string indent;
    std::string item_sep = ", ", key_sep = ": ";
    bool sort_keys = false;
};

inline void json_string(const std::string& s, bool ensure_ascii, std::string& out) {
    char buf[16];
    out += '"';
    size_t len = 0;
    for (size_t i = 0; i < s.size(); i += len) {
        const uint32_t c = decode_at(s, i, len);
        switch (c) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        if (c < 0x20) { std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c)); out += buf; }
        else if (c < 0x80 || !ensure_ascii) out.append(s, i, len);
        else if (c > 0xFFFF) {
            const uint32_t v = c - 0x10000;
            std::snprintf(buf, sizeof buf, "\\u%04x\\u%04x", unsigned(0xD800 + (v >> 10)), unsigned(0xDC00 + (v & 0x3FF)));
            out += buf;
        } else { std::snprintf(buf, sizeof buf, "\\u%04x", unsigned(c)); out += buf; }
    }
    out += '"';
}

inline void json_dump(const Value& v, const JsonStyle& style, size_t level, std::string& out) {
    const auto newline = [&](size_t depth) {
        if (!style.indented) return;
        out += '\n';
        for (size_t n = 0; n < depth; ++n) {
            out += style.indent;
            if (out.size() > kMaxBuilt) type_error("tojson built too long a string");
        }
    };
    if (out.size() > kMaxBuilt) type_error("tojson built too long a string");
    switch (v.k) {
        case Value::NONE: out += "null"; return;
        case Value::BOOL: out += v.b ? "true" : "false"; return;
        case Value::INT: out += std::to_string(v.i); return;
        case Value::FLOAT:
            out += std::isnan(v.f) ? "NaN" : std::isinf(v.f) ? (v.f > 0 ? "Infinity" : "-Infinity") : float_repr(v.f);
            return;
        case Value::STR: json_string(v.s, style.ensure_ascii, out); return;
        case Value::LIST: case Value::TUPLE: {
            if (v.seq->empty()) { out += "[]"; return; }
            out += '[';
            for (size_t n = 0; n < v.seq->size(); ++n) {
                if (n) out += style.item_sep;
                newline(level + 1);
                json_dump((*v.seq)[n], style, level + 1, out);
            }
            newline(level);
            out += ']';
            return;
        }
        case Value::DICT: {
            if (v.obj->items.empty()) { out += "{}"; return; }
            std::vector<const std::pair<std::string, Value>*> items;
            for (const auto& kv : v.obj->items) items.push_back(&kv);
            if (style.sort_keys)
                std::stable_sort(items.begin(), items.end(), [](const auto* a, const auto* b) { return a->first < b->first; });
            out += '{';
            for (size_t n = 0; n < items.size(); ++n) {
                if (n) out += style.item_sep;
                newline(level + 1);
                json_string(items[n]->first, style.ensure_ascii, out);
                out += style.key_sep;
                json_dump(items[n]->second, style, level + 1, out);
            }
            newline(level);
            out += '}';
            return;
        }
        default: type_error(std::string("Object of type ") + type_name(v) + " is not JSON serializable");
    }
}

// ---- the template language ----

// What one render reads and counts: the variables it was given, how deeply its statements and expressions nest at the moment, macro calls included, and the loop iterations and calls it has run.
struct Render {
    const Object* context;
    int depth = 0;
    uint64_t steps = 0;
    void step() { if (++steps > kMaxSteps) type_error("the render ran more loop iterations and macro calls than a chat prompt needs"); }
};

// One level of a render's nesting, counted while it lives; a render nesting past kMaxRenderDepth fails, as Python's recursion limit fails the reference.
struct Deeper {
    Render& r;
    explicit Deeper(Render& x) : r(x) {
        if (++r.depth > kMaxRenderDepth) { --r.depth; type_error("maximum recursion depth exceeded"); }
    }
    ~Deeper() { --r.depth; }
};

// Variables a template sets, each scope over the one it reads through: the root, a loop's iteration, a macro's call, a block set's body.
struct Scope {
    std::vector<std::pair<std::string, Value>> vars;
    std::shared_ptr<Scope> parent;
    void set(const std::string& name, Value v) {
        for (auto& kv : vars) if (kv.first == name) { kv.second = std::move(v); return; }
        vars.emplace_back(name, std::move(v));
    }
};

struct Env {
    std::shared_ptr<Scope> scope;
    Render* render;
};

// The globals of Jinja and of the reference renderer; the ones this renderer has no call for are refused when a template calls them.
inline bool global_name(const std::string& n) {
    return one_of(n, { "range", "namespace", "raise_exception", "strftime_now", "dict", "lipsum", "cycler", "joiner" });
}

inline Value lookup(const Env& env, const std::string& name) {
    for (const Scope* s = env.scope.get(); s; s = s->parent.get())
        for (const auto& kv : s->vars) if (kv.first == name) return kv.second;
    if (const Value* v = env.render->context->find(name)) return *v;
    if (global_name(name)) return Value::func(name);
    return undefined_name(name);
}

struct Expr {
    virtual ~Expr() = default;
    // Evaluates the expression one level deeper in the render.
    Value eval(Env& env) const { const Deeper d(*env.render); return run(env); }
    virtual Value run(Env& env) const = 0;
};
using ExprP = std::unique_ptr<Expr>;
using Kwargs = std::vector<std::pair<std::string, Value>>;
using KwExprs = std::vector<std::pair<std::string, ExprP>>;

// How a statement ends: on to the next, or a loop's break or continue.
enum class Flow { next, stop, skip };
struct Node {
    virtual ~Node() = default;
    // Runs the statement one level deeper in the render.
    Flow exec(Env& env, std::string& out) const { const Deeper d(*env.render); return run(env, out); }
    virtual Flow run(Env& env, std::string& out) const = 0;
};
using NodeP = std::unique_ptr<Node>;
using Body = std::vector<NodeP>;

inline Flow exec_body(const Body& body, Env& env, std::string& out) {
    for (const auto& n : body) {
        const Flow f = n->exec(env, out);
        if (f != Flow::next) return f;
    }
    return Flow::next;
}

// A call's arguments, positional and by keyword, read the way Python binds them to a function's parameters.
struct Args {
    const std::vector<Value>& pos;
    const Kwargs& kw;
    const char* what;
    // Parameter `index` named `name`, or null when neither passed it.
    const Value* get(size_t index, const char* name) const {
        const Value* by_name = nullptr;
        for (const auto& kv : kw) if (kv.first == name) by_name = &kv.second;
        if (index < pos.size()) {
            if (by_name) type_error(std::string(what) + "() got multiple values for argument '" + name + "'");
            return &pos[index];
        }
        return by_name;
    }
    void at_most(size_t n) const {
        if (pos.size() > n) type_error(std::string(what) + "() takes at most " + std::to_string(n) + " arguments (" + std::to_string(pos.size()) + " given)");
    }
};

inline Value call_value(const Value& f, const std::vector<Value>& args, const Kwargs& kwargs, Env& env);
inline Value apply_filter(const std::string& name, const Value& v, const std::vector<Value>& args, const Kwargs& kw, Env& env);
inline bool apply_test(const std::string& name, const Value& v, const std::vector<Value>& args, const Kwargs& kw);

// ---- expressions ----

struct Const : Expr {
    Value v;
    explicit Const(Value x) : v(std::move(x)) {}
    Value run(Env&) const override { return v; }
};

struct Name : Expr {
    std::string name;
    explicit Name(std::string n) : name(std::move(n)) {}
    Value run(Env& env) const override { return lookup(env, name); }
};

struct Getattr : Expr {
    ExprP obj;
    std::string attr;
    Getattr(ExprP o, std::string a) : obj(std::move(o)), attr(std::move(a)) {}
    Value run(Env& env) const override { return get_attr(obj->eval(env), attr); }
};

// `a:b:c` inside a subscript, whose parts may each be absent.
struct SliceExpr : Expr {
    ExprP start, stop, step;
    Value run(Env&) const override { type_error("a slice outside a subscript"); }
};

struct Getitem : Expr {
    ExprP obj, index;
    Getitem(ExprP o, ExprP i) : obj(std::move(o)), index(std::move(i)) {}
    Value run(Env& env) const override {
        const Value o = obj->eval(env);
        const auto* slice = dynamic_cast<const SliceExpr*>(index.get());
        if (!slice) return get_item(o, index->eval(env));
        const auto part = [&](const ExprP& e) { return e ? e->eval(env) : Value::none(); };
        const Value a = part(slice->start), b = part(slice->stop), c = part(slice->step);
        if (o.k == Value::UNDEF) undefined_error(o);
        if (o.k == Value::STR) {
            const auto at = char_offsets(o.s);
            const SliceRange r = slice_range(int64_t(at.size()) - 1, a, b, c);
            std::string out;
            for (int64_t n = 0, x = r.start; n < r.count; ++n, x += r.step) out.append(o.s, at[size_t(x)], at[size_t(x) + 1] - at[size_t(x)]);
            return Value::str(out);
        }
        if (!o.sequence()) return Value::undefined(object_type_repr(o) + " has no element slice(" + repr(a) + ", " + repr(b) + ", " + repr(c) + ")");
        const SliceRange r = slice_range(int64_t(o.seq->size()), a, b, c);
        std::vector<Value> out;
        for (int64_t n = 0, x = r.start; n < r.count; ++n, x += r.step) out.push_back((*o.seq)[size_t(x)]);
        return Value::list(std::move(out), o.k == Value::TUPLE);
    }
};

// Python's integers have no bound; past 64 bits the render fails rather than wrap.
[[noreturn]] inline void overflow() { type_error("an integer past 64 bits is not supported"); }
constexpr int64_t kIntMax = std::numeric_limits<int64_t>::max(), kIntMin = std::numeric_limits<int64_t>::min();

inline int64_t add_int(int64_t a, int64_t b) {
    if ((b > 0 && a > kIntMax - b) || (b < 0 && a < kIntMin - b)) overflow();
    return a + b;
}
inline int64_t sub_int(int64_t a, int64_t b) {
    if ((b < 0 && a > kIntMax + b) || (b > 0 && a < kIntMin + b)) overflow();
    return a - b;
}
inline int64_t mul_int(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a == -1) { if (b == kIntMin) overflow(); return -b; }
    if (b == -1) { if (a == kIntMin) overflow(); return -a; }
    const bool over = a > 0 ? (b > 0 ? a > kIntMax / b : b < kIntMin / a) : (b > 0 ? a < kIntMin / b : a < kIntMax / b);
    if (over) overflow();
    return a * b;
}
// Python's floor division and modulo, whose results take the divisor's sign.
inline int64_t floor_div(int64_t a, int64_t b) {
    if (a == kIntMin && b == -1) overflow();
    int64_t q = a / b;
    if (a % b != 0 && ((a < 0) != (b < 0))) --q;
    return q;
}
inline int64_t floor_mod(int64_t a, int64_t b) {
    if (b == -1) return 0;
    const int64_t r = a % b;
    return r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}
// The same for floats, as Python computes them: the remainder from fmod, the quotient from the difference, then rounded to the nearest whole number, so 1 // 0.1 is 9.0.
inline double float_mod(double x, double y) {
    double r = std::fmod(x, y);
    if (r != 0) { if ((y < 0) != (r < 0)) r += y; }
    else r = std::copysign(0.0, y);
    return r;
}
inline double float_floor_div(double x, double y) {
    const double r = std::fmod(x, y);
    double q = (x - r) / y;
    if (r != 0 && (y < 0) != (r < 0)) q -= 1.0;
    if (q == 0) return std::copysign(0.0, x / y);
    double whole = std::floor(q);
    if (q - whole > 0.5) whole += 1.0;
    return whole;
}

struct Unary : Expr {
    char op;  // '-', '+' or '!' for not
    ExprP e;
    Unary(char o, ExprP x) : op(o), e(std::move(x)) {}
    Value run(Env& env) const override {
        const Value v = e->eval(env);
        if (op == '!') return Value::boolean(!truthy(v));
        if (v.k == Value::UNDEF) undefined_error(v);
        if (!v.numeric()) type_error(std::string("bad operand type for unary ") + op + ": '" + type_name(v) + "'");
        if (v.k == Value::FLOAT) return Value::number(op == '-' ? -v.f : v.f);
        return Value::integer(op == '-' ? sub_int(0, v.as_int()) : v.as_int());
    }
};

// A string or sequence repeated `times` times, as Python's * repeats one.
inline Value repeat(const Value& x, int64_t times) {
    const size_t unit = x.k == Value::STR ? x.s.size() : x.seq->size();
    if (times <= 0 || unit == 0) return x.k == Value::STR ? Value::str("") : Value::list({}, x.k == Value::TUPLE);
    if (uint64_t(times) > kMaxBuilt / unit) type_error(std::string("a repeat built too long a ") + type_name(x));
    if (x.k == Value::STR) { std::string out; for (int64_t k = 0; k < times; ++k) out += x.s; return Value::str(out); }
    std::vector<Value> items;
    for (int64_t k = 0; k < times; ++k) items.insert(items.end(), x.seq->begin(), x.seq->end());
    return Value::list(std::move(items), x.k == Value::TUPLE);
}

inline Value arithmetic(const std::string& op, const Value& a, const Value& b) {
    if (a.k == Value::UNDEF) undefined_error(a);
    if (b.k == Value::UNDEF) undefined_error(b);
    if (a.numeric() && b.numeric()) {
        const bool real = a.k == Value::FLOAT || b.k == Value::FLOAT;
        const double x = a.as_float(), y = b.as_float();
        const int64_t m = a.as_int(), n = b.as_int();
        if ((op == "/" || op == "//" || op == "%") && (real ? y == 0 : n == 0)) {
            if (op == "/") type_error(real ? "float division by zero" : "division by zero");
            if (op == "//") type_error(real ? "float floor division by zero" : "integer division or modulo by zero");
            type_error(real ? "float modulo" : "integer modulo by zero");
        }
        if (op == "/") return Value::number(x / y);
        if (real) {
            if (op == "+") return Value::number(x + y);
            if (op == "-") return Value::number(x - y);
            if (op == "*") return Value::number(x * y);
            if (op == "//") return Value::number(float_floor_div(x, y));
            if (op == "%") return Value::number(float_mod(x, y));
            if (x == 0 && y < 0) type_error("0.0 cannot be raised to a negative power");
            if (x < 0 && y != std::floor(y)) type_error("a negative number raised to a fractional power is complex");
            const double r = std::pow(x, y);
            if (std::isinf(r) && std::isfinite(x) && std::isfinite(y)) type_error("(34, 'Numerical result out of range')");
            return Value::number(r);
        }
        if (op == "+") return Value::integer(add_int(m, n));
        if (op == "-") return Value::integer(sub_int(m, n));
        if (op == "*") return Value::integer(mul_int(m, n));
        if (op == "//") return Value::integer(floor_div(m, n));
        if (op == "%") return Value::integer(floor_mod(m, n));
        if (n < 0) {
            if (m == 0) type_error("0.0 cannot be raised to a negative power");
            return Value::number(std::pow(x, y));
        }
        int64_t r = 1, base = m;
        for (int64_t e = n; e > 0; e >>= 1) {
            if (e & 1) r = mul_int(r, base);
            if (e > 1) base = mul_int(base, base);
        }
        return Value::integer(r);
    }
    if (op == "+" && a.k == b.k && (a.k == Value::STR || a.sequence())) {
        if ((a.k == Value::STR ? a.s.size() + b.s.size() : a.seq->size() + b.seq->size()) > kMaxBuilt) type_error("a + built too long a value");
        if (a.k == Value::STR) return Value::str(a.s + b.s);
        std::vector<Value> items = *a.seq;
        items.insert(items.end(), b.seq->begin(), b.seq->end());
        return Value::list(std::move(items), a.k == Value::TUPLE);
    }
    // Python names the left operand's own type in these, where its sequence refuses the other operand.
    const auto repeatable = [](const Value& v) { return v.k == Value::STR || v.sequence(); };
    if (op == "+" && repeatable(a))
        type_error(std::string("can only concatenate ") + type_name(a) + " (not \"" + type_name(b) + "\") to " + type_name(a));
    if (op == "*" && (repeatable(a) || repeatable(b))) {
        if (a.integral() || b.integral()) return a.integral() ? repeat(b, a.as_int()) : repeat(a, b.as_int());
        type_error(std::string("can't multiply sequence by non-int of type '") + type_name(repeatable(a) ? b : a) + "'");
    }
    if (op == "%" && a.k == Value::STR) type_error("printf-style string formatting is not supported");
    if (op == "-" && (a.set_like() || b.set_like())) type_error("set operations on dict views are not supported");
    type_error("unsupported operand type(s) for " + op + ": '" + type_name(a) + "' and '" + type_name(b) + "'");
}

struct BinOp : Expr {
    std::string op;  // + - * / // % **
    ExprP l, r;
    BinOp(std::string o, ExprP a, ExprP b) : op(std::move(o)), l(std::move(a)), r(std::move(b)) {}
    Value run(Env& env) const override {
        const Value a = l->eval(env);
        return arithmetic(op, a, r->eval(env));
    }
};

// `~`, which joins its operands' str().
struct Concat : Expr {
    std::vector<ExprP> parts;
    Value run(Env& env) const override {
        std::string out;
        for (const auto& p : parts) {
            out += to_str(p->eval(env));
            if (out.size() > kMaxBuilt) type_error("a ~ built too long a string");
        }
        return Value::str(out);
    }
};

// `and` and `or`, which give one of their operands and read the second only when it decides.
struct AndOr : Expr {
    bool is_and;
    ExprP l, r;
    AndOr(bool a, ExprP x, ExprP y) : is_and(a), l(std::move(x)), r(std::move(y)) {}
    Value run(Env& env) const override {
        Value a = l->eval(env);
        if (truthy(a) != is_and) return a;
        return r->eval(env);
    }
};

struct Compare : Expr {
    ExprP first;
    std::vector<std::pair<std::string, ExprP>> ops;  // == != < > <= >= in notin
    Value run(Env& env) const override {
        Value a = first->eval(env);
        for (const auto& op : ops) {
            Value b = op.second->eval(env);
            bool r;
            const std::string& o = op.first;
            if (o == "==") r = equal(a, b);
            else if (o == "!=") r = !equal(a, b);
            else if (o == "<" || o == ">" || o == "<=" || o == ">=") r = compare(o, a, b);
            else if (o == "in") r = contains(b, a);
            else r = !contains(b, a);
            if (!r) return Value::boolean(false);
            a = std::move(b);
        }
        return Value::boolean(true);
    }
};

struct CondExpr : Expr {
    ExprP test, yes, no;  // `no` may be absent, and then the result is undefined
    int line = 1;         // the template line the expression starts on, which that undefined value's message names
    Value run(Env& env) const override {
        if (truthy(test->eval(env))) return yes->eval(env);
        if (no) return no->eval(env);
        return Value::undefined("the inline if-expression on line " + std::to_string(line) + " evaluated to false and no else section was defined.");
    }
};

struct ListLit : Expr {
    std::vector<ExprP> items;
    bool tuple = false;
    Value run(Env& env) const override {
        std::vector<Value> out;
        for (const auto& e : items) out.push_back(e->eval(env));
        return Value::list(std::move(out), tuple);
    }
};

struct DictLit : Expr {
    std::vector<std::pair<ExprP, ExprP>> items;
    Value run(Env& env) const override {
        auto o = std::make_shared<Object>();
        for (const auto& kv : items) {
            const Value key = kv.first->eval(env);
            if (key.k != Value::STR) type_error("a dict key here must be a string");
            o->set(key.s, kv.second->eval(env));
        }
        return Value::dict(o);
    }
};

inline void eval_args(const std::vector<ExprP>& args, const KwExprs& kwargs, Env& env, std::vector<Value>& a, Kwargs& kw) {
    for (const auto& e : args) a.push_back(e->eval(env));
    for (const auto& kv : kwargs) kw.emplace_back(kv.first, kv.second->eval(env));
}

struct Call : Expr {
    ExprP callee;
    std::vector<ExprP> args;
    KwExprs kwargs;
    Value run(Env& env) const override {
        const Value f = callee->eval(env);
        std::vector<Value> a;
        Kwargs kw;
        eval_args(args, kwargs, env, a, kw);
        return call_value(f, a, kw, env);
    }
};

struct FilterExpr : Expr {
    ExprP value;
    std::string name;
    std::vector<ExprP> args;
    KwExprs kwargs;
    Value run(Env& env) const override {
        const Value v = value->eval(env);
        std::vector<Value> a;
        Kwargs kw;
        eval_args(args, kwargs, env, a, kw);
        return apply_filter(name, v, a, kw, env);
    }
};

struct TestExpr : Expr {
    ExprP value;
    std::string name;
    std::vector<ExprP> args;
    KwExprs kwargs;
    Value run(Env& env) const override {
        const Value v = value->eval(env);
        std::vector<Value> a;
        Kwargs kw;
        eval_args(args, kwargs, env, a, kw);
        return Value::boolean(apply_test(name, v, a, kw));
    }
};

// A filter or test Jinja does not have, inside an if or a conditional expression, where Jinja compiles it and fails only if it runs.
struct Missing : Expr {
    ExprP value;
    std::string message;
    Value run(Env& env) const override {
        value->eval(env);
        throw TemplateError(message);
    }
};

// ---- statements ----

// What `for` and `set` assign to: names, several of them unpacking a sequence, or a namespace's attribute.
struct Target {
    std::vector<std::string> names;
    bool unpack = false;
    std::string attr;  // set on names[0], a namespace
};

inline void assign(const Target& t, const Value& v, Env& env) {
    if (!t.attr.empty()) {
        const Value ns = lookup(env, t.names[0]);
        if (ns.k != Value::NS) type_error("cannot assign attribute on non-namespace object");
        if (v.holds_ns) type_error("a namespace holding a namespace is not supported");
        if (v.depth + 1 > kMaxValueDepth) nested_too_deeply();
        ns.obj->set(t.attr, v);
        return;
    }
    if (!t.unpack) { env.scope->set(t.names[0], v); return; }
    const std::vector<Value> items = iterate(v);
    if (items.size() > t.names.size()) type_error("too many values to unpack (expected " + std::to_string(t.names.size()) + ")");
    if (items.size() < t.names.size())
        type_error("not enough values to unpack (expected " + std::to_string(t.names.size()) + ", got " + std::to_string(items.size()) + ")");
    for (size_t n = 0; n < items.size(); ++n) env.scope->set(t.names[n], items[n]);
}

inline void emit(std::string& out, const std::string& text) {
    out += text;
    if (out.size() > kMaxBuilt) type_error("the render grew longer than a chat prompt can be");
}

struct TextNode : Node {
    std::string text;
    explicit TextNode(std::string t) : text(std::move(t)) {}
    Flow run(Env&, std::string& out) const override { emit(out, text); return Flow::next; }
};

struct OutputNode : Node {
    ExprP e;
    explicit OutputNode(ExprP x) : e(std::move(x)) {}
    Flow run(Env& env, std::string& out) const override { emit(out, to_str(e->eval(env))); return Flow::next; }
};

struct IfNode : Node {
    std::vector<std::pair<ExprP, Body>> branches;
    Body otherwise;
    Flow run(Env& env, std::string& out) const override {
        for (const auto& b : branches) if (truthy(b.first->eval(env))) return exec_body(b.second, env, out);
        return exec_body(otherwise, env, out);
    }
};

inline Env child(const Env& env) {
    auto s = std::make_shared<Scope>();
    s->parent = env.scope;
    return Env{ s, env.render };
}

// A loop: each iteration sets its target and `loop` in a scope of its own, so what it sets is gone after it, as in Jinja.
struct ForNode : Node {
    Target target;
    ExprP iter, filter;
    Body body, otherwise;
    Flow run(Env& env, std::string& out) const override {
        std::vector<Value> items = iterate(iter->eval(env));
        if (filter) {
            std::vector<Value> kept;
            for (auto& item : items) {
                env.render->step();
                Env e = child(env);
                assign(target, item, e);
                if (truthy(filter->eval(e))) kept.push_back(std::move(item));
            }
            items = std::move(kept);
        }
        // The else runs outside this loop, so a break or continue in it belongs to the loop around it.
        if (items.empty()) {
            Env e = child(env);
            return exec_body(otherwise, e, out);
        }
        const int64_t n = int64_t(items.size());
        for (int64_t k = 0; k < n; ++k) {
            env.render->step();
            Env e = child(env);
            assign(target, items[size_t(k)], e);
            auto loop = std::make_shared<Object>();
            loop->set("index", Value::integer(k + 1));
            loop->set("index0", Value::integer(k));
            loop->set("revindex", Value::integer(n - k));
            loop->set("revindex0", Value::integer(n - k - 1));
            loop->set("first", Value::boolean(k == 0));
            loop->set("last", Value::boolean(k + 1 == n));
            loop->set("length", Value::integer(n));
            loop->set("depth", Value::integer(1));
            loop->set("depth0", Value::integer(0));
            loop->set("previtem", k > 0 ? items[size_t(k) - 1] : Value::undefined("there is no previous item"));
            loop->set("nextitem", k + 1 < n ? items[size_t(k) + 1] : Value::undefined("there is no next item"));
            e.scope->set("loop", Value::dict(loop, Value::LOOP));
            if (exec_body(body, e, out) == Flow::stop) break;
        }
        return Flow::next;
    }
};

struct SetNode : Node {
    Target target;
    ExprP value;
    Flow run(Env& env, std::string&) const override { assign(target, value->eval(env), env); return Flow::next; }
};

struct FilterCall {
    std::string name;
    std::vector<ExprP> args;
    KwExprs kwargs;
};

// `{% set x %}...{% endset %}`: the body's text, rendered in a scope of its own, through the filters after the name.
struct SetBlockNode : Node {
    Target target;
    std::vector<FilterCall> filters;
    Body body;
    Flow run(Env& env, std::string&) const override {
        std::string text;
        Env e = child(env);
        // The body runs inline, so a break or continue in it leaves the loop around it before anything is assigned.
        if (const Flow f = exec_body(body, e, text); f != Flow::next) return f;
        Value v = Value::str(text);
        for (const auto& f : filters) {
            std::vector<Value> a;
            Kwargs kw;
            eval_args(f.args, f.kwargs, e, a, kw);
            v = apply_filter(f.name, v, a, kw, e);
        }
        assign(target, v, env);
        return Flow::next;
    }
};

// The reference renderer's `{% generation %}` block, which marks the assistant's text for its token mask and renders its body as it is, in a scope of its own.
struct GenerationNode : Node {
    Body body;
    Flow run(Env& env, std::string& out) const override {
        Env e = child(env);
        exec_body(body, e, out);
        return Flow::next;
    }
};

struct MacroNode : Node {
    std::string name;
    std::vector<std::string> params;
    std::vector<ExprP> defaults;  // one per parameter, absent where it has none
    Body body;
    Flow run(Env& env, std::string&) const override {
        Value v;
        v.k = Value::MACRO;
        v.s = name;
        v.macro = std::make_shared<const Closure>(Closure{ this, env.scope });
        env.scope->set(name, v);
        return Flow::next;
    }
};

struct LoopControl : Node {
    Flow flow;
    explicit LoopControl(Flow f) : flow(f) {}
    Flow run(Env&, std::string&) const override { return flow; }
};

inline Value call_macro(const Value& m, const std::vector<Value>& args, const Kwargs& kwargs, Env& env) {
    const MacroNode& node = *m.macro->node;
    const std::shared_ptr<Scope> scope = m.macro->scope.lock();
    if (!scope) type_error("macro " + str_repr(node.name) + " called after its scope ended");
    if (args.size() > node.params.size())
        type_error("macro " + str_repr(node.name) + " takes not more than " + std::to_string(node.params.size()) + " argument(s)");
    for (const auto& kv : kwargs) {
        const auto at = std::find(node.params.begin(), node.params.end(), kv.first);
        if (at == node.params.end() || size_t(at - node.params.begin()) < args.size())
            type_error("macro " + str_repr(node.name) + " takes no keyword argument " + str_repr(kv.first));
    }
    env.render->step();
    auto s = std::make_shared<Scope>();
    s->parent = scope;
    Env e{ s, env.render };
    std::string out;
    for (size_t n = 0; n < node.params.size(); ++n) {
        const std::string& p = node.params[n];
        if (n < args.size()) { s->set(p, args[n]); continue; }
        const auto kw = std::find_if(kwargs.begin(), kwargs.end(), [&](const auto& kv) { return kv.first == p; });
        if (kw != kwargs.end()) s->set(p, kw->second);
        else s->set(p, node.defaults[n] ? node.defaults[n]->eval(e) : Value::undefined("parameter " + str_repr(p) + " was not provided"));
    }
    exec_body(node.body, e, out);
    return Value::str(out);
}

// `range` holds at most this many items, as in Jinja's sandbox.
constexpr int64_t kMaxRange = 100000;

inline Value call_global(const std::string& name, const std::vector<Value>& args, const Kwargs& kwargs) {
    const Args a{ args, kwargs, name.c_str() };
    if (name == "raise_exception") {
        const Value* message = a.get(0, "message");
        if (!message || args.size() + kwargs.size() != 1) type_error("raise_exception() takes one argument");
        throw TemplateError(to_str(*message));
    }
    if (name == "namespace") {
        auto o = std::make_shared<Object>();
        a.at_most(1);
        if (!args.empty()) {
            if (args[0].k != Value::DICT) type_error("namespace() takes a mapping");
            o->items = args[0].obj->items;
        }
        for (const auto& kv : kwargs) o->set(kv.first, kv.second);
        return Value::dict(o, Value::NS);
    }
    if (name == "range") {
        if (!kwargs.empty() || args.empty() || args.size() > 3) type_error("range expected 1 to 3 integer arguments");
        for (const auto& v : args) if (!v.integral()) type_error(std::string("'") + type_name(v) + "' object cannot be interpreted as an integer");
        const int64_t start = args.size() > 1 ? args[0].as_int() : 0, stop = args.size() > 1 ? args[1].as_int() : args[0].as_int();
        const int64_t step = args.size() > 2 ? args[2].as_int() : 1;
        if (step == 0) type_error("range() arg 3 must not be zero");
        // The distance in unsigned arithmetic, which holds it whatever the signs.
        uint64_t count = 0;
        if (step > 0 && stop > start) count = (uint64_t(stop) - uint64_t(start) - 1) / uint64_t(step) + 1;
        if (step < 0 && start > stop) count = (uint64_t(start) - uint64_t(stop) - 1) / (uint64_t(0) - uint64_t(step)) + 1;
        if (count > uint64_t(kMaxRange)) type_error("Range too big. The sandbox blocks ranges larger than MAX_RANGE (100000).");
        std::vector<Value> out;
        for (uint64_t k = 0; k < count; ++k) out.push_back(Value::integer(start + int64_t(k) * step));
        return Value::list(std::move(out));
    }
    if (name == "strftime_now") {
        const Value* format = a.get(0, "format");
        if (!format || format->k != Value::STR || args.size() + kwargs.size() != 1) type_error("strftime_now takes a format string");
        if (format->s.find('\0') != std::string::npos) type_error("embedded null character");
        const std::time_t now = std::time(nullptr);
        std::tm local{};
#if defined(_WIN32)
        localtime_s(&local, &now);
#else
        localtime_r(&now, &local);
#endif
        char buf[512];
        size_t n = 0;
        if (!format->s.empty()) {
#if defined(_MSC_VER)
            // The MSVC runtime ends the process on a conversion it does not know unless the thread has a handler of its own; with one that returns, strftime gives 0 and the render fails, as Python's strftime fails on Windows.
            const _invalid_parameter_handler previous = _set_thread_local_invalid_parameter_handler([](const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {});
            n = std::strftime(buf, sizeof buf, format->s.c_str(), &local);
            _set_thread_local_invalid_parameter_handler(previous);
#else
            n = std::strftime(buf, sizeof buf, format->s.c_str(), &local);
#endif
            if (n == 0) type_error("strftime_now's format is not supported here, or gives an empty or too long date");
        }
        return Value::str(std::string(buf, n));
    }
    type_error("the global " + str_repr(name) + " is not supported");
}

inline Value call_method(const std::string& name, const Value& self, const std::vector<Value>& args, const Kwargs& kwargs) {
    const Args a{ args, kwargs, name.c_str() };
    if (self.k == Value::DICT) {
        if (name == "get") {
            a.at_most(2);
            const Value* key = a.get(0, "key");
            const Value* fallback = a.get(1, "default");
            if (!key) type_error("get expected at least 1 argument, got 0");
            if (key->k == Value::STR) if (const Value* v = self.obj->find(key->s)) return *v;
            return fallback ? *fallback : Value::none();
        }
        std::vector<Value> out;
        if (name == "keys" || name == "values" || name == "items") {
            if (!kwargs.empty()) type_error("dict." + name + "() takes no keyword arguments");
            if (!args.empty()) type_error("dict." + name + "() takes no arguments (" + std::to_string(args.size()) + " given)");
            for (const auto& kv : self.obj->items) {
                if (name == "keys") out.push_back(Value::str(kv.first));
                else if (name == "values") out.push_back(kv.second);
                else out.push_back(Value::list({ Value::str(kv.first), kv.second }, true));
            }
            return Value::view("dict_" + name, std::move(out));
        }
    }
    if (self.sequence() && name == "count") {
        if (args.size() != 1 || !kwargs.empty()) type_error("count() takes exactly one argument");
        int64_t n = 0;
        for (const auto& x : *self.seq) n += equal(x, args[0]);
        return Value::integer(n);
    }
    if (self.k == Value::STR) {
        const std::string& s = self.s;
        const auto text = [&](size_t index, const char* param) -> const std::string* {
            const Value* v = a.get(index, param);
            if (!v || v->k == Value::NONE) return nullptr;
            if (v->k != Value::STR) type_error(name + " arg must be None or str");
            return &v->s;
        };
        if (name == "strip" || name == "lstrip" || name == "rstrip") {
            a.at_most(1);
            return Value::str(py_strip(s, text(0, "chars"), name != "rstrip", name != "lstrip"));
        }
        if (name == "split") {
            a.at_most(2);
            const Value* limit = a.get(1, "maxsplit");
            if (limit && !limit->integral()) type_error(std::string("'") + type_name(*limit) + "' object cannot be interpreted as an integer");
            std::vector<Value> out;
            for (auto& part : py_split(s, text(0, "sep"), limit ? limit->as_int() : -1)) out.push_back(Value::str(std::move(part)));
            return Value::list(std::move(out));
        }
        if (name == "startswith" || name == "endswith") {
            const Value* affix = args.size() == 1 && kwargs.empty() ? &args[0] : nullptr;
            if (!affix) type_error(name + "() takes exactly one argument here");
            std::vector<std::string> options;
            if (affix->k == Value::STR) options.push_back(affix->s);
            else if (affix->k == Value::TUPLE) for (const auto& v : *affix->seq) {
                if (v.k != Value::STR) type_error("tuple for " + name + " must only contain str, not " + type_name(v));
                options.push_back(v.s);
            } else type_error(name + " first arg must be str or a tuple of str, not " + type_name(*affix));
            for (const auto& o : options) {
                if (o.size() > s.size()) continue;
                if (name == "startswith" ? s.compare(0, o.size(), o) == 0 : s.compare(s.size() - o.size(), o.size(), o) == 0) return Value::boolean(true);
            }
            return Value::boolean(false);
        }
        if (name == "replace") {
            a.at_most(3);
            const std::string* old = text(0, "old");
            const std::string* with = text(1, "new");
            const Value* count = a.get(2, "count");
            if (!old || !with) type_error("replace expected at least 2 arguments");
            if (count && !count->integral()) type_error(std::string("'") + type_name(*count) + "' object cannot be interpreted as an integer");
            return Value::str(py_replace(s, *old, *with, count ? count->as_int() : -1));
        }
        if (name == "lower" || name == "upper" || name == "title") {
            if (!args.empty() || !kwargs.empty()) type_error(name + "() takes no arguments");
            return Value::str(name == "title" ? py_title(s) : ascii_case(s, name == "upper", name));
        }
        if (name == "count" || name == "find") {
            a.at_most(1);
            const std::string* sub = text(0, "sub");
            if (!sub) type_error(name + "() takes a string");
            if (name == "find") {
                const size_t hit = s.find(*sub);
                return Value::integer(hit == std::string::npos ? -1 : int64_t(char_count(s.substr(0, hit))));
            }
            if (sub->empty()) return Value::integer(int64_t(char_count(s)) + 1);
            int64_t n = 0;
            for (size_t pos = s.find(*sub); pos != std::string::npos; pos = s.find(*sub, pos + sub->size())) ++n;
            return Value::integer(n);
        }
        if (name == "join") {
            if (args.size() != 1 || !kwargs.empty()) type_error("join() takes exactly one argument");
            std::string out;
            bool first = true;
            for (const auto& v : iterate(args[0])) {
                if (v.k != Value::STR) type_error(std::string("sequence item: expected str instance, ") + type_name(v) + " found");
                if (!first) out += s;
                out += v.s;
                first = false;
                if (out.size() > kMaxBuilt) type_error("a join built too long a string");
            }
            return Value::str(out);
        }
    }
    type_error(std::string("'") + type_name(self) + "' object has no supported method " + str_repr(name));
}

inline Value call_value(const Value& f, const std::vector<Value>& args, const Kwargs& kwargs, Env& env) {
    switch (f.k) {
        case Value::MACRO: return call_macro(f, args, kwargs, env);
        case Value::FUNC: return call_global(f.s, args, kwargs);
        case Value::METHOD: return call_method(f.s, (*f.seq)[0], args, kwargs);
        case Value::UNDEF: undefined_error(f);
        default: type_error(std::string("'") + type_name(f) + "' object is not callable");
    }
}

// ---- filters and tests ----

inline bool builtin_filter(const std::string& n) {
    return one_of(n, { "abs", "attr", "batch", "capitalize", "center", "count", "d", "default", "dictsort", "e", "escape",
                       "filesizeformat", "first", "float", "forceescape", "format", "groupby", "indent", "int", "items",
                       "join", "last", "length", "list", "lower", "map", "max", "min", "pprint", "random", "reject",
                       "rejectattr", "replace", "reverse", "round", "safe", "select", "selectattr", "slice", "sort",
                       "string", "striptags", "sum", "title", "tojson", "trim", "truncate", "unique", "upper",
                       "urlencode", "urlize", "wordcount", "wordwrap", "xmlattr" });
}
inline bool supported_filter(const std::string& n) {
    return one_of(n, { "count", "d", "default", "dictsort", "first", "items", "join", "last", "length", "list", "lower",
                       "map", "reject", "rejectattr", "replace", "reverse", "safe", "select", "selectattr", "string", "title",
                       "tojson", "trim", "upper" });
}
// The filters that keep the items a test passes or fails, and the argument that names the test: the second after an attribute's name.
inline bool select_filter(const std::string& n) { return one_of(n, { "select", "reject", "selectattr", "rejectattr" }); }
inline size_t test_argument(const std::string& filter) { return one_of(filter, { "selectattr", "rejectattr" }) ? 1 : 0; }
// The keyword arguments each supported filter takes here.
inline bool filter_keyword(const std::string& filter, const std::string& kw) {
    if (filter == "d" || filter == "default") return one_of(kw, { "default_value", "boolean" });
    if (filter == "dictsort") return one_of(kw, { "case_sensitive", "by", "reverse" });
    if (filter == "join") return kw == "d";
    if (filter == "map") return one_of(kw, { "attribute", "default" });
    if (filter == "replace") return one_of(kw, { "old", "new", "count" });
    if (filter == "tojson") return one_of(kw, { "ensure_ascii", "indent", "separators", "sort_keys" });
    if (filter == "trim") return kw == "chars";
    return false;
}
inline bool builtin_test(const std::string& n) {
    return one_of(n, { "boolean", "callable", "defined", "divisibleby", "eq", "equalto", "escaped", "even", "false",
                       "filter", "float", "ge", "greaterthan", "gt", "in", "integer", "iterable", "le", "lessthan",
                       "lower", "lt", "mapping", "ne", "none", "number", "odd", "sameas", "sequence", "string", "test",
                       "true", "undefined", "upper" });
}
// Every test Jinja has but four: sameas, whose identity Python gives small numbers and strings alike, and escaped, filter and test, which ask about the environment.
inline bool supported_test(const std::string& n) { return builtin_test(n) && !one_of(n, { "escaped", "filter", "test", "sameas" }); }
// The methods a template may call.
inline bool supported_method(const std::string& n) {
    return one_of(n, { "startswith", "endswith", "strip", "lstrip", "rstrip", "split", "replace", "lower", "upper", "title",
                       "count", "find", "join", "get", "keys", "values", "items" });
}

// An attribute path as Jinja's attribute getter reads it: dotted parts, each an index when it is all digits.
inline std::vector<Value> attribute_path(const Value& attribute) {
    std::vector<Value> path;
    if (attribute.k != Value::STR) return { attribute };
    const std::string dot = ".";
    for (const auto& part : py_split(attribute.s, &dot, -1)) {
        const bool digits = !part.empty() && part.size() < 19 && part.find_first_not_of("0123456789") == std::string::npos;
        path.push_back(digits ? Value::integer(std::stoll(part)) : Value::str(part));
    }
    return path;
}

inline std::vector<Value> dict_items(const Value& v) {
    std::vector<Value> out;
    for (const auto& kv : v.obj->items) out.push_back(Value::list({ Value::str(kv.first), kv.second }, true));
    return out;
}

// The map filter with its positional arguments from `at` on: a filter name and that filter's arguments, or none beside the `attribute` keyword.
// Each map is one level deeper in the render, and a map naming map reads the same arguments one further on, so a chain of such names fails at the render's depth limit rather than recursing on the host stack.
inline Value map_filter(const Value& v, const std::vector<Value>& args, size_t at, const Kwargs& kw, Env& env) {
    const Deeper d(*env.render);
    const Args a{ args, kw, "map" };
    std::vector<Value> out;
    if (const Value* attribute = a.get(size_t(-1), "attribute")) {
        if (at < args.size()) type_error("map takes a filter name or an attribute, not both");
        const Value* fallback = a.get(size_t(-1), "default");
        const std::vector<Value> path = attribute_path(*attribute);
        for (const auto& item : iterate(v)) {
            Value x = item;
            for (const auto& part : path) {
                x = get_item(x, part);
                if (x.k == Value::UNDEF && fallback && fallback->k != Value::NONE) x = *fallback;
            }
            out.push_back(x);
        }
        return Value::list(std::move(out));
    }
    if (at >= args.size() || args[at].k != Value::STR) type_error("map requires a filter argument");
    const std::string& filter = args[at].s;
    if (!supported_filter(filter)) type_error("the filter " + str_repr(filter) + " is not supported");
    if (filter == "map") {
        for (const auto& item : iterate(v)) out.push_back(map_filter(item, args, at + 1, kw, env));
        return Value::list(std::move(out));
    }
    const std::vector<Value> rest(args.begin() + at + 1, args.end());
    for (const auto& item : iterate(v)) out.push_back(apply_filter(filter, item, rest, kw, env));
    return Value::list(std::move(out));
}

inline Value apply_filter(const std::string& name, const Value& v, const std::vector<Value>& args, const Kwargs& kw, Env& env) {
    const Args a{ args, kw, name.c_str() };
    if (name == "default" || name == "d") {
        const Value* fallback = a.get(0, "default_value");
        const Value* boolean = a.get(1, "boolean");
        if (v.k == Value::UNDEF || (boolean && truthy(*boolean) && !truthy(v))) return fallback ? *fallback : Value::str("");
        return v;
    }
    if (name == "length" || name == "count") return Value::integer(length(v));
    if (name == "string" || name == "safe") return Value::str(to_str(v));
    if (name == "trim") {
        const Value* chars = a.get(0, "chars");
        if (chars && chars->k != Value::NONE && chars->k != Value::STR) type_error("strip arg must be None or str");
        return Value::str(py_strip(to_str(v), chars && chars->k == Value::STR ? &chars->s : nullptr, true, true));
    }
    if (name == "upper" || name == "lower") return Value::str(ascii_case(to_str(v), name == "upper", name));
    if (name == "title") return Value::str(title_filter(to_str(v)));
    if (name == "tojson") {
        JsonStyle style;
        if (const Value* x = a.get(0, "ensure_ascii")) style.ensure_ascii = truthy(*x);
        if (const Value* x = a.get(1, "indent"); x && x->k != Value::NONE) {
            style.indented = true;
            if (x->integral()) {
                if (x->as_int() > int64_t(kMaxBuilt)) type_error("tojson built too long a string");
                style.indent = std::string(size_t(std::max<int64_t>(x->as_int(), 0)), ' ');
            }
            else if (x->k == Value::STR) style.indent = x->s;
            else type_error("tojson's indent must be an integer or a string");
            style.item_sep = ",";
        }
        if (const Value* x = a.get(2, "separators"); x && x->k != Value::NONE) {
            if (!x->sequence() || x->seq->size() != 2 || (*x->seq)[0].k != Value::STR || (*x->seq)[1].k != Value::STR)
                type_error("tojson's separators must be two strings");
            style.item_sep = (*x->seq)[0].s;
            style.key_sep = (*x->seq)[1].s;
        }
        if (const Value* x = a.get(3, "sort_keys")) style.sort_keys = truthy(*x);
        std::string out;
        json_dump(v, style, 0, out);
        return Value::str(out);
    }
    if (name == "items") {
        if (v.k == Value::UNDEF) return Value::list({});
        if (v.k != Value::DICT) type_error("Can only get item pairs from a mapping.");
        return Value::list(dict_items(v));
    }
    if (name == "first" || name == "last") {
        std::vector<Value> items = iterate(v);
        if (items.empty()) return Value::undefined(name == "first" ? "No first item, sequence was empty." : "No last item, sequence was empty.");
        return name == "first" ? items.front() : items.back();
    }
    if (name == "list") return Value::list(iterate(v));
    if (name == "reverse") {
        if (v.k == Value::STR) {
            const auto at = char_offsets(v.s);
            std::string out;
            for (size_t n = at.size() - 1; n > 0; --n) out.append(v.s, at[n - 1], at[n] - at[n - 1]);
            return Value::str(out);
        }
        std::vector<Value> items = iterate(v);
        std::reverse(items.begin(), items.end());
        return Value::list(std::move(items));
    }
    if (name == "join") {
        const Value* sep = a.get(0, "d");
        const std::string glue = sep ? to_str(*sep) : std::string();
        std::string out;
        bool first = true;
        for (const auto& item : iterate(v)) {
            if (!first) out += glue;
            out += to_str(item);
            first = false;
            if (out.size() > kMaxBuilt) type_error("a join built too long a string");
        }
        return Value::str(out);
    }
    if (name == "replace") {
        const Value* old = a.get(0, "old");
        const Value* with = a.get(1, "new");
        const Value* count = a.get(2, "count");
        if (!old || !with) type_error("do_replace() missing a required argument");
        if (count && count->k != Value::NONE && !count->integral()) type_error(std::string("'") + type_name(*count) + "' object cannot be interpreted as an integer");
        return Value::str(py_replace(to_str(v), to_str(*old), to_str(*with), count && count->k != Value::NONE ? count->as_int() : -1));
    }
    if (name == "dictsort") {
        if (v.k != Value::DICT) type_error(std::string("'") + type_name(v) + "' object has no attribute 'items'");
        const Value* sensitive = a.get(0, "case_sensitive");
        const Value* by = a.get(1, "by");
        const Value* reverse = a.get(2, "reverse");
        const std::string field = by ? to_str(*by) : "key";
        if (field != "key" && field != "value") type_error("You can only sort by either \"key\" or \"value\"");
        const std::vector<Value> items = dict_items(v);
        const bool fold = !(sensitive && truthy(*sensitive));
        std::vector<Value> keys;
        for (const auto& pair : items) {
            const Value& x = (*pair.seq)[field == "key" ? 0 : 1];
            keys.push_back(x.k == Value::STR && fold ? Value::str(ascii_case(x.s, false, "dictsort")) : x);
        }
        std::vector<size_t> order(items.size());
        for (size_t n = 0; n < order.size(); ++n) order[n] = n;
        // Python's sort with reverse keeps equal items in their order, as a stable sort on the swapped comparison does.
        const bool backwards = reverse && truthy(*reverse);
        std::stable_sort(order.begin(), order.end(), [&](size_t p, size_t q) { return backwards ? less(keys[q], keys[p]) : less(keys[p], keys[q]); });
        std::vector<Value> out;
        for (size_t n : order) out.push_back(items[n]);
        return Value::list(std::move(out));
    }
    if (name == "map") return map_filter(v, args, 0, kw, env);
    if (select_filter(name)) {
        // The items whose value, or attribute, passes the named test, or failing it for reject; with no test named, those that are true.
        const bool keep = name[0] == 's';
        const size_t at = test_argument(name);
        if (at && args.empty()) type_error("Missing parameter for attribute name");
        if (!truthy(v)) return Value::list({});
        const std::vector<Value> path = at ? attribute_path(args[0]) : std::vector<Value>();
        std::string test;
        if (args.size() > at) {
            const Value& named = args[at];
            if (named.k == Value::UNDEF) type_error("No test named Undefined. (" + named.s + "; did you forget to quote the callable name?)");
            if (named.k != Value::STR || !builtin_test(named.s)) type_error("No test named " + repr(named) + ".");
            if (!supported_test(named.s)) type_error("the test " + str_repr(named.s) + " is not supported");
            test = named.s;
        }
        const std::vector<Value> rest(args.begin() + std::min(args.size(), at + 1), args.end());
        std::vector<Value> out;
        for (const auto& item : iterate(v)) {
            Value x = item;
            for (const auto& part : path) x = get_item(x, part);
            if ((test.empty() ? truthy(x) : apply_test(test, x, rest, kw)) == keep) out.push_back(item);
        }
        return Value::list(std::move(out));
    }
    type_error("the filter " + str_repr(name) + " is not supported");
}

inline bool apply_test(const std::string& name, const Value& v, const std::vector<Value>& args, const Kwargs& kw) {
    const Args a{ args, kw, name.c_str() };
    const auto other = [&]() -> Value {
        const Value* x = a.get(0, "other");
        if (!x) type_error("the test " + str_repr(name) + " takes one argument");
        return *x;
    };
    if (name == "defined") return v.k != Value::UNDEF;
    if (name == "undefined") return v.k == Value::UNDEF;
    if (name == "none") return v.k == Value::NONE;
    if (name == "true") return v.k == Value::BOOL && v.b;
    if (name == "false") return v.k == Value::BOOL && !v.b;
    if (name == "boolean") return v.k == Value::BOOL;
    if (name == "string") return v.k == Value::STR;
    if (name == "number") return v.numeric();
    if (name == "integer") return v.k == Value::INT;
    if (name == "float") return v.k == Value::FLOAT;
    if (name == "mapping") return v.k == Value::DICT;
    if (name == "iterable") return v.k == Value::UNDEF || v.k == Value::STR || v.sequence() || v.k == Value::DICT || v.k == Value::VIEW || v.k == Value::LOOP;
    if (name == "sequence") return v.k == Value::UNDEF || v.k == Value::STR || v.sequence() || v.k == Value::DICT;
    if (name == "callable") return v.k == Value::UNDEF || v.k == Value::LOOP || v.k == Value::MACRO || v.k == Value::FUNC || v.k == Value::METHOD;
    if (name == "lower" || name == "upper") {
        const std::string s = to_str(v);
        ascii_only(s, "a case test");
        bool cased = false;
        for (char c : s) {
            if (ascii_letter(c)) cased = true;
            if (name == "lower" ? (c >= 'A' && c <= 'Z') : (c >= 'a' && c <= 'z')) return false;
        }
        return cased;
    }
    if (name == "odd" || name == "even" || name == "divisibleby") {
        const Value d = name == "divisibleby" ? other() : Value::integer(2);
        if (!v.integral() || !d.integral()) type_error("the test " + str_repr(name) + " takes integers here");
        if (d.as_int() == 0) type_error("integer modulo by zero");
        const int64_t r = floor_mod(v.as_int(), d.as_int());
        return name == "odd" ? r == 1 : r == 0;
    }
    if (name == "eq" || name == "equalto") return equal(v, other());
    if (name == "ne") return !equal(v, other());
    if (name == "lt" || name == "lessthan") return compare("<", v, other());
    if (name == "gt" || name == "greaterthan") return compare(">", v, other());
    if (name == "le") return compare("<=", v, other());
    if (name == "ge") return compare(">=", v, other());
    if (name == "in") return contains(other(), v);
    type_error("the test " + str_repr(name) + " is not supported");
}

// ---- lexer ----

struct Token {
    enum Type : uint8_t { DATA, BLOCK_BEGIN, BLOCK_END, VAR_BEGIN, VAR_END, NAME, STRING, INTEGER, FLOAT, OP, END } type;
    std::string text;  // DATA's text, a name, a string's value, an operator
    int64_t integer = 0;
    double number = 0;
    int line = 1;      // the template line a token inside a tag starts on
};

[[noreturn]] inline void refuse(const std::string& why) { throw Refused(why); }

// A string literal's body read as Jinja reads it: its non-ASCII characters written as escapes, then Python's unicode-escape decoding of the whole.
inline std::string unescape(const std::string& body) {
    std::string out;
    const auto hex = [&](size_t at, size_t digits) {
        uint32_t v = 0;
        if (at + digits > body.size()) refuse("a truncated escape in a string literal");
        for (size_t n = 0; n < digits; ++n) {
            const char c = body[at + n];
            if (!std::isxdigit((unsigned char)c)) refuse("a truncated escape in a string literal");
            v = v * 16 + uint32_t(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10);
        }
        return v;
    };
    for (size_t i = 0; i < body.size(); ++i) {
        if (body[i] != '\\' || i + 1 == body.size()) { out += body[i]; continue; }
        const char e = body[++i];
        switch (e) {
            case '\n': break;
            case '\\': out += '\\'; break;
            case '\'': out += '\''; break;
            case '"': out += '"'; break;
            case 'a': out += '\a'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'v': out += '\v'; break;
            case 'x': out += utf8::encode(hex(i + 1, 2)); i += 2; break;
            case 'u': out += utf8::encode(hex(i + 1, 4)); i += 4; break;
            case 'U': { const uint32_t cp = hex(i + 1, 8); if (cp > 0x10FFFF) refuse("an escape past U+10FFFF"); out += utf8::encode(cp); i += 8; break; }
            case 'N': refuse("named escapes in string literals are not supported");
            default:
                if (e >= '0' && e <= '7') {
                    uint32_t v = 0;
                    size_t n = 0;
                    for (; n < 3 && i + n < body.size() && body[i + n] >= '0' && body[i + n] <= '7'; ++n) v = v * 8 + uint32_t(body[i + n] - '0');
                    out += utf8::encode(v);
                    i += n - 1;
                } else if (static_cast<unsigned char>(e) >= 0x80) {
                    // A backslash before a non-ASCII character stays, and the character reads back as its own escape text.
                    size_t len = 0;
                    const uint32_t c = decode_at(body, i, len);
                    char buf[16];
                    std::snprintf(buf, sizeof buf, c <= 0xFF ? "\\x%02x" : c <= 0xFFFF ? "\\u%04x" : "\\U%08x", unsigned(c));
                    out += buf;
                    i += len - 1;
                } else { out += '\\'; out += e; }
        }
    }
    return out;
}

// Whitespace as Jinja's \s reads it, from s[i]; its length in bytes, or 0.
inline size_t space_at(const std::string& s, size_t i) {
    size_t len = 0;
    return i < s.size() && is_space(decode_at(s, i, len)) ? len : 0;
}

// The template as tokens, with Jinja's whitespace rules applied to the text between tags: trim_blocks, lstrip_blocks, and the - and + markers.
inline std::vector<Token> tokenize(const std::string& source) {
    // Newlines read as \n, and one newline ending the template is dropped.
    std::string src;
    for (size_t i = 0; i < source.size(); ++i) {
        if (source[i] == '\r') { src += '\n'; if (i + 1 < source.size() && source[i + 1] == '\n') ++i; }
        else src += source[i];
    }
    if (!src.empty() && src.back() == '\n') src.pop_back();

    std::vector<Token> out;
    const size_t n = src.size();
    size_t pos = 0;
    // Whether the text before the next tag starts a line, which lstrip_blocks needs when that text holds no newline.
    bool line_start = true;
    const auto starts = [&](size_t at, const char* s) { return src.compare(at, std::strlen(s), s) == 0; };
    // The line of src[at], counted as the tokens are read in order.
    size_t counted = 0;
    int lines = 1;
    const auto line_at = [&](size_t at) { for (; counted < at; ++counted) lines += src[counted] == '\n'; return lines; };
    const auto skip_space = [&](size_t at) { for (size_t len; (len = space_at(src, at)); ) at += len; return at; };
    while (pos < n) {
        size_t open = std::string::npos;
        for (size_t at = src.find('{', pos); at != std::string::npos && at + 1 < n; at = src.find('{', at + 1))
            if (src[at + 1] == '{' || src[at + 1] == '%' || src[at + 1] == '#') { open = at; break; }
        std::string text = src.substr(pos, open == std::string::npos ? std::string::npos : open - pos);
        if (open == std::string::npos) {
            if (!text.empty()) out.push_back({ Token::DATA, text });
            break;
        }
        const char kind = src[open + 1];
        const char sign = open + 2 < n && (src[open + 2] == '-' || src[open + 2] == '+') ? src[open + 2] : 0;
        if (sign == '-') text = py_strip(text, nullptr, false, true);
        else if (sign != '+' && kind != '{') {
            // lstrip_blocks: whitespace alone between a line's start and a block or comment tag is dropped.
            const size_t newline = text.rfind('\n');
            const size_t line = newline == std::string::npos ? 0 : newline + 1;
            if (line > 0 || line_start) {
                size_t at = line;
                for (size_t len; (len = space_at(text, at)); ) at += len;
                if (at == text.size()) text.erase(line);
            }
        }
        if (!text.empty()) out.push_back({ Token::DATA, text });
        pos = open + 2 + (sign ? 1 : 0);
        if (kind == '#') {
            const size_t close = src.find("#}", pos);
            if (close == std::string::npos) refuse("a comment without its end");
            size_t after = close + 2;
            const char end_sign = close > pos ? src[close - 1] : 0;
            if (end_sign == '-') after = skip_space(after);
            else if (end_sign != '+' && after < n && src[after] == '\n') ++after;
            line_start = src[after - 1] == '\n';
            pos = after;
            continue;
        }
        out.push_back({ kind == '%' ? Token::BLOCK_BEGIN : Token::VAR_BEGIN, {} });
        std::string brackets;
        for (;;) {
            pos = skip_space(pos);
            if (pos >= n) refuse(kind == '%' ? "a block tag without its end" : "an expression without its end");
            if (brackets.empty()) {
                size_t end = 0;
                bool strip = false, trim = false;
                if (kind == '%') {
                    if (starts(pos, "-%}")) { end = 3; strip = true; }
                    else if (starts(pos, "+%}")) end = 3;
                    else if (starts(pos, "%}")) { end = 2; trim = true; }
                } else {
                    if (starts(pos, "-}}")) { end = 3; strip = true; }
                    else if (starts(pos, "}}")) end = 2;
                }
                if (end) {
                    pos += end;
                    if (strip) pos = skip_space(pos);
                    else if (trim && pos < n && src[pos] == '\n') ++pos;
                    line_start = src[pos - 1] == '\n';
                    out.push_back({ kind == '%' ? Token::BLOCK_END : Token::VAR_END, {} });
                    break;
                }
            }
            const char c = src[pos];
            const int line = line_at(pos);
            if (std::isalpha((unsigned char)c) || c == '_') {
                size_t e = pos;
                while (e < n && (std::isalnum((unsigned char)src[e]) || src[e] == '_')) ++e;
                out.push_back({ Token::NAME, src.substr(pos, e - pos) });
                pos = e;
            } else if (std::isdigit((unsigned char)c)) {
                const auto digits = [&](size_t at) {
                    size_t e = at;
                    while (e < n && (std::isdigit((unsigned char)src[e]) || (src[e] == '_' && e > at && e + 1 < n && std::isdigit((unsigned char)src[e + 1])))) ++e;
                    return e;
                };
                size_t e = digits(pos);
                bool real = false;
                // After a dot only an integer is read, so `a.0.1` is two subscripts.
                if (!(pos > 0 && src[pos - 1] == '.')) {
                    if (e + 1 < n && src[e] == '.' && std::isdigit((unsigned char)src[e + 1])) { e = digits(e + 1); real = true; }
                    if (e < n && (src[e] == 'e' || src[e] == 'E')) {
                        size_t x = e + 1;
                        if (x < n && (src[x] == '+' || src[x] == '-')) ++x;
                        if (x < n && std::isdigit((unsigned char)src[x])) { e = digits(x); real = true; }
                    }
                }
                std::string literal;
                for (size_t k = pos; k < e; ++k) if (src[k] != '_') literal += src[k];
                // Jinja reads a leading zero as its own literal, so digits after it are a syntax error there.
                if (!real && literal.size() > 1 && literal[0] == '0' && literal.find_first_not_of('0') != std::string::npos)
                    refuse("an integer literal with a leading zero");
                Token t{ real ? Token::FLOAT : Token::INTEGER, literal };
                if (real) t.number = std::strtod(literal.c_str(), nullptr);
                else {
                    errno = 0;
                    t.integer = std::strtoll(literal.c_str(), nullptr, 10);
                    if (errno) refuse("an integer literal past 64 bits");
                }
                out.push_back(t);
                pos = e;
            } else if (c == '\'' || c == '"') {
                size_t e = pos + 1;
                while (e < n && src[e] != c) e += src[e] == '\\' ? 2 : 1;
                if (e >= n) refuse("a string literal without its end");
                out.push_back({ Token::STRING, unescape(src.substr(pos + 1, e - pos - 1)) });
                pos = e + 1;
            } else {
                static const char* const ops[] = { "**", "//", "==", "!=", ">=", "<=", "+", "-", "/", "*", "%", "~", "[", "]",
                                                   "(", ")", "{", "}", "<", ">", "=", ".", ":", "|", ",", ";" };
                const char* op = nullptr;
                for (const char* o : ops) if (starts(pos, o)) { op = o; break; }
                if (!op) refuse("an unexpected character " + str_repr(std::string(1, c)));
                if (op[0] == '(' || op[0] == '[' || op[0] == '{') brackets += op[0];
                else if (op[0] == ')' || op[0] == ']' || op[0] == '}') {
                    const char want = op[0] == ')' ? '(' : op[0] == ']' ? '[' : '{';
                    if (brackets.empty() || brackets.back() != want) refuse(std::string("an unbalanced '") + op + "'");
                    brackets.pop_back();
                }
                out.push_back({ Token::OP, op });
                pos += std::strlen(op);
            }
            out.back().line = line;
        }
    }
    out.push_back({ Token::END, {} });
    return out;
}

// ---- parser ----

// Jinja's grammar, over the tokens; what the renderer does not support is refused here, before any render.
class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : t_(std::move(tokens)) {}

    Body parse() {
        Body body = subparse({});
        for (const auto& m : missing_) if (!m.second) refuse(m.first);
        return body;
    }

    // The names the template reads as an attribute, a constant subscript, a get() or a filter's attribute, and the constant separators it splits text at.
    std::set<std::string> fields, separators;

private:
    std::vector<Token> t_;
    size_t p_ = 0;
    int loops_ = 0;       // enclosing loops, which break and continue need
    int fors_ = 0;        // enclosing for tags, across macros, inside which Jinja refuses to assign to `loop`
    bool soft_ = false;   // inside an if or a conditional expression, where Jinja fails on an unknown filter or test only if it runs
    bool macro_ = false;  // inside a macro's body
    int depth_ = 0;
    std::vector<std::pair<std::string, bool>> missing_;  // each filter or test Jinja does not have, and whether it was in a soft place

    struct Nest {
        Parser& p;
        explicit Nest(Parser& q) : p(q) { if (++p.depth_ > kMaxNest) refuse("nested too deeply"); }
        ~Nest() { --p.depth_; }
    };
    // A chain the parser builds in a loop, such as a + b + c or a.b.c, nests the tree one level deeper at each link, so each link counts as a nested call does.
    struct Chain {
        Parser& p;
        int links = 0;
        explicit Chain(Parser& q) : p(q) {}
        void link() { ++links; if (++p.depth_ > kMaxNest) refuse("nested too deeply"); }
        ~Chain() { p.depth_ -= links; }
    };
    template <class T> struct Keep {
        T& ref;
        T saved;
        Keep(T& r, T v) : ref(r), saved(r) { ref = v; }
        ~Keep() { ref = saved; }
    };

    const Token& cur() const { return t_[p_]; }
    const Token& look() const { return t_[std::min(p_ + 1, t_.size() - 1)]; }
    bool is_name(const char* v) const { return cur().type == Token::NAME && cur().text == v; }
    bool is_op(const char* v) const { return cur().type == Token::OP && cur().text == v; }
    void next() { if (p_ + 1 < t_.size()) ++p_; }
    [[noreturn]] void unexpected() const {
        const Token& t = cur();
        std::string what = t.type == Token::END ? "the end of the template" : t.type == Token::BLOCK_END ? "the end of a block tag"
                         : t.type == Token::VAR_END ? "the end of an expression" : "'" + t.text + "'";
        refuse("unexpected " + what);
    }
    void expect(Token::Type type) { if (cur().type != type) unexpected(); next(); }
    void expect_op(const char* v) { if (!is_op(v)) unexpected(); next(); }
    void expect_word(const char* v) { if (!is_name(v)) unexpected(); next(); }
    std::string expect_name() {
        if (cur().type != Token::NAME) unexpected();
        std::string s = cur().text;
        next();
        return s;
    }

    Body subparse(std::initializer_list<const char*> ends) {
        Body body;
        for (;;) {
            const Token& t = cur();
            if (t.type == Token::END) {
                if (ends.size()) refuse(std::string("a block without its '") + *ends.begin() + "'");
                return body;
            }
            if (t.type == Token::DATA) { body.push_back(std::make_unique<TextNode>(t.text)); next(); continue; }
            if (t.type == Token::VAR_BEGIN) {
                next();
                body.push_back(std::make_unique<OutputNode>(parse_tuple(true)));
                expect(Token::VAR_END);
                continue;
            }
            if (t.type != Token::BLOCK_BEGIN) unexpected();
            next();
            if (cur().type != Token::NAME) unexpected();
            for (const char* e : ends) if (cur().text == e) return body;
            body.push_back(statement());
            expect(Token::BLOCK_END);
        }
    }

    // A tag's header ends, then its body runs to one of `ends`, which is consumed when `drop`.
    Body statements(std::initializer_list<const char*> ends, bool drop) {
        if (is_op(":")) next();
        expect(Token::BLOCK_END);
        Body body = subparse(ends);
        if (drop) next();
        return body;
    }

    NodeP statement() {
        Nest nest(*this);
        const std::string tag = expect_name();
        if (tag == "if") return parse_if();
        if (tag == "for") return parse_for();
        if (tag == "set") return parse_set();
        if (tag == "macro") return parse_macro();
        if (tag == "generation") {
            // The reference renderer compiles the body as a call block's caller, a frame of its own.
            auto node = std::make_unique<GenerationNode>();
            Keep<bool> soft(soft_, false);
            Keep<int> loops(loops_, 0);
            node->body = statements({ "endgeneration" }, true);
            return node;
        }
        if (tag == "break" || tag == "continue") {
            if (!loops_) refuse("'" + tag + "' outside a loop");
            return std::make_unique<LoopControl>(tag == "break" ? Flow::stop : Flow::skip);
        }
        refuse("the tag '" + tag + "' is not supported");
    }

    NodeP parse_if() {
        auto node = std::make_unique<IfNode>();
        Keep<bool> soft(soft_, true);
        for (;;) {
            ExprP test = parse_tuple(false);
            Body body = statements({ "elif", "else", "endif" }, false);
            node->branches.emplace_back(std::move(test), std::move(body));
            const std::string t = expect_name();
            if (t == "elif") continue;
            if (t == "else") node->otherwise = statements({ "endif" }, true);
            break;
        }
        return node;
    }

    Target parse_target(bool with_namespace) {
        Target t;
        if (with_namespace && cur().type == Token::NAME && look().type == Token::OP && look().text == ".") {
            t.names.push_back(expect_name());
            next();
            t.attr = expect_name();
            return t;
        }
        const bool paren = is_op("(");
        if (paren) next();
        for (;;) {
            const std::string name = expect_name();
            if (one_of(name, { "true", "false", "none", "True", "False", "None" })) refuse("cannot assign to '" + name + "'");
            if (fors_ && name == "loop") refuse("Can't assign to special loop variable in for-loop target");
            t.names.push_back(name);
            if (!is_op(",")) break;
            next();
            t.unpack = true;
            if (is_name("in") || is_op("=") || is_op(")") || cur().type == Token::BLOCK_END) break;
        }
        if (paren) expect_op(")");
        return t;
    }

    NodeP parse_for() {
        auto node = std::make_unique<ForNode>();
        Keep<int> fors(fors_, fors_ + 1);
        node->target = parse_target(false);
        expect_word("in");
        node->iter = parse_tuple(false, { "recursive" });
        // The filter, the body and the else each run in a frame of their own, where an unknown filter fails when the template is compiled.
        Keep<bool> soft(soft_, false);
        if (is_name("if")) {
            next();
            node->filter = parse_expression(true);
        }
        if (is_name("recursive")) refuse("recursive loops are not supported");
        {
            Keep<int> loops(loops_, loops_ + 1);
            node->body = statements({ "endfor", "else" }, false);
        }
        if (expect_name() == "else") node->otherwise = statements({ "endfor" }, true);
        return node;
    }

    NodeP parse_set() {
        Target target = parse_target(true);
        if (is_op("=")) {
            next();
            auto node = std::make_unique<SetNode>();
            node->target = std::move(target);
            node->value = parse_tuple(true);
            return node;
        }
        if (target.unpack || !target.attr.empty()) refuse("a block set assigns one name");
        auto node = std::make_unique<SetBlockNode>();
        node->target = std::move(target);
        Keep<bool> soft(soft_, false);
        while (is_op("|")) {
            next();
            FilterCall f;
            f.name = dotted_name();
            if (is_op("(")) parse_call_args(f.args, f.kwargs);
            if (!check_filter(f.name, f.args, f.kwargs, true)) refuse("the filter '" + f.name + "' does not exist");
            node->filters.push_back(std::move(f));
        }
        node->body = statements({ "endset" }, true);
        return node;
    }

    NodeP parse_macro() {
        auto node = std::make_unique<MacroNode>();
        node->name = expect_name();
        expect_op("(");
        Keep<bool> soft(soft_, false);
        Keep<bool> in_macro(macro_, true);
        Keep<int> loops(loops_, 0);
        bool defaults = false;
        while (!is_op(")")) {
            if (!node->params.empty()) expect_op(",");
            const std::string param = expect_name();
            if (one_of(param, { "true", "false", "none", "True", "False", "None" })) refuse("cannot assign to '" + param + "'");
            if (std::find(node->params.begin(), node->params.end(), param) != node->params.end())
                refuse("duplicate argument '" + param + "' in function definition");
            node->params.push_back(param);
            if (is_op("=")) { next(); node->defaults.push_back(parse_expression(true)); defaults = true; }
            else if (defaults) refuse("a macro parameter without a default follows one with a default");
            else node->defaults.push_back(nullptr);
        }
        next();
        node->body = statements({ "endmacro" }, true);
        return node;
    }

    bool tuple_end(std::initializer_list<const char*> extra) const {
        if (cur().type == Token::VAR_END || cur().type == Token::BLOCK_END || is_op(")")) return true;
        for (const char* e : extra) if (is_name(e)) return true;
        return false;
    }

    ExprP parse_tuple(bool condexpr, std::initializer_list<const char*> extra = {}, bool parens = false) {
        std::vector<ExprP> items;
        bool tuple = false;
        for (;;) {
            if (!items.empty()) expect_op(",");
            if (tuple_end(extra)) break;
            items.push_back(parse_expression(condexpr));
            if (is_op(",")) tuple = true;
            else break;
        }
        if (!tuple) {
            if (!items.empty()) return std::move(items[0]);
            if (!parens) unexpected();
        }
        auto t = std::make_unique<ListLit>();
        t->items = std::move(items);
        t->tuple = true;
        return t;
    }

    ExprP parse_expression(bool condexpr) { return condexpr ? parse_condexpr() : parse_or(); }

    ExprP parse_condexpr() {
        Nest nest(*this);
        const size_t mark = missing_.size();
        int line = cur().line;
        ExprP e = parse_or();
        Chain chain(*this);
        while (is_name("if")) {
            chain.link();
            next();
            // Jinja compiles the whole conditional expression in a soft frame, the part before `if` included.
            for (size_t k = mark; k < missing_.size(); ++k) missing_[k].second = true;
            Keep<bool> soft(soft_, true);
            auto c = std::make_unique<CondExpr>();
            c->line = line;
            c->yes = std::move(e);
            c->test = parse_or();
            if (is_name("else")) { next(); c->no = parse_condexpr(); }
            e = std::move(c);
            line = cur().line;
        }
        return e;
    }

    ExprP parse_or() {
        ExprP l = parse_and();
        Chain chain(*this);
        while (is_name("or")) { chain.link(); next(); l = std::make_unique<AndOr>(false, std::move(l), parse_and()); }
        return l;
    }
    ExprP parse_and() {
        ExprP l = parse_not();
        Chain chain(*this);
        while (is_name("and")) { chain.link(); next(); l = std::make_unique<AndOr>(true, std::move(l), parse_not()); }
        return l;
    }
    ExprP parse_not() {
        Nest nest(*this);
        if (is_name("not")) { next(); return std::make_unique<Unary>('!', parse_not()); }
        return parse_compare();
    }
    ExprP parse_compare() {
        ExprP e = parse_math1();
        auto c = std::make_unique<Compare>();
        for (;;) {
            if (is_op("==") || is_op("!=") || is_op("<") || is_op(">") || is_op("<=") || is_op(">=")) {
                const std::string op = cur().text;
                next();
                c->ops.emplace_back(op, parse_math1());
            } else if (is_name("in")) {
                next();
                c->ops.emplace_back("in", parse_math1());
            } else if (is_name("not") && look().type == Token::NAME && look().text == "in") {
                next(); next();
                c->ops.emplace_back("notin", parse_math1());
            } else break;
        }
        if (c->ops.empty()) return e;
        c->first = std::move(e);
        return c;
    }
    ExprP parse_math1() {
        ExprP l = parse_concat();
        Chain chain(*this);
        while (is_op("+") || is_op("-")) {
            chain.link();
            const std::string op = cur().text;
            next();
            l = std::make_unique<BinOp>(op, std::move(l), parse_concat());
        }
        return l;
    }
    ExprP parse_concat() {
        ExprP first = parse_math2();
        if (!is_op("~")) return first;
        auto c = std::make_unique<Concat>();
        c->parts.push_back(std::move(first));
        while (is_op("~")) { next(); c->parts.push_back(parse_math2()); }
        return c;
    }
    ExprP parse_math2() {
        ExprP l = parse_pow();
        Chain chain(*this);
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            chain.link();
            const std::string op = cur().text;
            next();
            l = std::make_unique<BinOp>(op, std::move(l), parse_pow());
        }
        return l;
    }
    ExprP parse_pow() {
        ExprP l = parse_unary(true);
        Chain chain(*this);
        while (is_op("**")) { chain.link(); next(); l = std::make_unique<BinOp>("**", std::move(l), parse_unary(true)); }
        return l;
    }
    ExprP parse_unary(bool with_filter) {
        Nest nest(*this);
        ExprP node;
        if (is_op("-") || is_op("+")) {
            const char op = cur().text[0];
            next();
            node = std::make_unique<Unary>(op, parse_unary(false));
        } else node = parse_primary();
        node = parse_postfix(std::move(node));
        if (with_filter) node = parse_filter_expr(std::move(node));
        return node;
    }

    ExprP parse_primary() {
        const Token& t = cur();
        if (t.type == Token::NAME) {
            const std::string name = t.text;
            next();
            if (name == "true" || name == "True") return std::make_unique<Const>(Value::boolean(true));
            if (name == "false" || name == "False") return std::make_unique<Const>(Value::boolean(false));
            if (name == "none" || name == "None") return std::make_unique<Const>(Value::none());
            if (macro_ && one_of(name, { "varargs", "kwargs", "caller" })) refuse("macros reading '" + name + "' are not supported");
            return std::make_unique<Name>(name);
        }
        if (t.type == Token::STRING) {
            std::string s;
            while (cur().type == Token::STRING) { s += cur().text; next(); }
            return std::make_unique<Const>(Value::str(s));
        }
        if (t.type == Token::INTEGER) { const int64_t v = t.integer; next(); return std::make_unique<Const>(Value::integer(v)); }
        if (t.type == Token::FLOAT) { const double v = t.number; next(); return std::make_unique<Const>(Value::number(v)); }
        if (is_op("(")) {
            next();
            ExprP e = parse_tuple(true, {}, true);
            expect_op(")");
            return e;
        }
        if (is_op("[")) {
            next();
            auto l = std::make_unique<ListLit>();
            while (!is_op("]")) {
                if (!l->items.empty()) expect_op(",");
                if (is_op("]")) break;
                l->items.push_back(parse_expression(true));
            }
            next();
            return l;
        }
        if (is_op("{")) {
            next();
            auto d = std::make_unique<DictLit>();
            while (!is_op("}")) {
                if (!d->items.empty()) expect_op(",");
                if (is_op("}")) break;
                ExprP key = parse_expression(true);
                expect_op(":");
                d->items.emplace_back(std::move(key), parse_expression(true));
            }
            next();
            return d;
        }
        unexpected();
    }

    ExprP parse_postfix(ExprP node) {
        Chain chain(*this);
        for (;;) {
            if (is_op(".") || is_op("[") || is_op("(")) chain.link();
            if (is_op(".")) {
                next();
                if (cur().type == Token::NAME) {
                    const std::string attr = expect_name();
                    fields.insert(attr);
                    node = std::make_unique<Getattr>(std::move(node), attr);
                } else if (cur().type == Token::INTEGER) {
                    const int64_t v = cur().integer;
                    next();
                    node = std::make_unique<Getitem>(std::move(node), std::make_unique<Const>(Value::integer(v)));
                } else unexpected();
            } else if (is_op("[")) {
                next();
                std::vector<ExprP> args;
                while (!is_op("]")) {
                    if (!args.empty()) expect_op(",");
                    args.push_back(parse_subscribed());
                }
                next();
                if (args.size() != 1) refuse("a subscript of several indexes is not supported");
                if (const auto* c = dynamic_cast<const Const*>(args[0].get()); c && c->v.k == Value::STR) fields.insert(c->v.s);
                node = std::make_unique<Getitem>(std::move(node), std::move(args[0]));
            } else if (is_op("(")) node = parse_call(std::move(node));
            else break;
        }
        return node;
    }

    ExprP parse_subscribed() {
        ExprP first;
        if (!is_op(":")) {
            first = parse_expression(true);
            if (!is_op(":")) return first;
        }
        next();
        auto s = std::make_unique<SliceExpr>();
        s->start = std::move(first);
        if (!is_op("]") && !is_op(",") && !is_op(":")) s->stop = parse_expression(true);
        if (is_op(":")) {
            next();
            if (!is_op("]") && !is_op(",")) s->step = parse_expression(true);
        }
        return s;
    }

    void parse_call_args(std::vector<ExprP>& args, KwExprs& kwargs) {
        expect_op("(");
        bool comma = false;
        while (!is_op(")")) {
            if (comma) { expect_op(","); if (is_op(")")) break; }
            if (is_op("*") || is_op("**")) refuse("unpacked call arguments are not supported");
            if (cur().type == Token::NAME && look().type == Token::OP && look().text == "=") {
                const std::string key = expect_name();
                for (const auto& kv : kwargs) if (kv.first == key) refuse("keyword argument repeated: " + key);
                next();
                kwargs.emplace_back(key, parse_expression(true));
            } else {
                if (!kwargs.empty()) refuse("a positional argument follows a keyword argument");
                args.push_back(parse_expression(true));
            }
            comma = true;
        }
        next();
    }

    ExprP parse_call(ExprP callee) {
        auto c = std::make_unique<Call>();
        const auto* attr = dynamic_cast<const Getattr*>(callee.get());
        if (attr && !supported_method(attr->attr)) refuse("the method '" + attr->attr + "' is not supported");
        if (const auto* name = dynamic_cast<const Name*>(callee.get()))
            if (one_of(name->name, { "dict", "lipsum", "cycler", "joiner" })) refuse("the global '" + name->name + "' is not supported");
        parse_call_args(c->args, c->kwargs);
        if (attr && !c->args.empty())
            if (const auto* first = dynamic_cast<const Const*>(c->args[0].get()); first && first->v.k == Value::STR) {
                if (attr->attr == "get") fields.insert(first->v.s);
                if (attr->attr == "split") separators.insert(first->v.s);
            }
        c->callee = std::move(callee);
        return c;
    }

    ExprP parse_filter_expr(ExprP node) {
        Chain chain(*this);
        for (;;) {
            if (is_op("|") || is_name("is") || is_op("(")) chain.link();
            if (is_op("|")) node = parse_filter(std::move(node));
            else if (is_name("is")) node = parse_test(std::move(node));
            else if (is_op("(")) node = parse_call(std::move(node));
            else break;
        }
        return node;
    }

    std::string dotted_name() {
        std::string name = expect_name();
        while (is_op(".")) { next(); name += "." + expect_name(); }
        return name;
    }

    // Whether a filter or test can be built: a supported one with the arguments it takes here, else refused, except one Jinja does not have, which a soft place defers to its run.
    bool check_filter(const std::string& name, const std::vector<ExprP>& args, const KwExprs& kwargs, bool filter) {
        const bool builtin = filter ? builtin_filter(name) : builtin_test(name);
        const bool supported = filter ? supported_filter(name) : supported_test(name);
        const std::string kind = filter ? "filter" : "test";
        if (!builtin) {
            missing_.emplace_back("the " + kind + " '" + name + "' does not exist", soft_);
            return false;
        }
        if (!supported) refuse("the " + kind + " '" + name + "' is not supported");
        for (const auto& kv : kwargs)
            if (!filter || !filter_keyword(name, kv.first)) refuse("the " + kind + " '" + name + "' does not take '" + kv.first + "' here");
        if (filter && name == "map" && !args.empty())
            if (const auto* c = dynamic_cast<const Const*>(args[0].get()); c && c->v.k == Value::STR && !supported_filter(c->v.s))
                refuse("the filter '" + c->v.s + "' is not supported");
        if (filter && select_filter(name) && args.size() > test_argument(name))
            if (const auto* c = dynamic_cast<const Const*>(args[test_argument(name)].get()); c && c->v.k == Value::STR && builtin_test(c->v.s) && !supported_test(c->v.s))
                refuse("the test '" + c->v.s + "' is not supported");
        // An attribute a filter reads by name is a field the template reads.
        const ExprP* attribute = nullptr;
        if (filter && (name == "selectattr" || name == "rejectattr") && !args.empty()) attribute = &args[0];
        if (filter && name == "map") for (const auto& kv : kwargs) if (kv.first == "attribute") attribute = &kv.second;
        if (attribute)
            if (const auto* c = dynamic_cast<const Const*>(attribute->get()); c && c->v.k == Value::STR)
                for (const auto& part : attribute_path(c->v)) if (part.k == Value::STR) fields.insert(part.s);
        return true;
    }

    ExprP parse_filter(ExprP node) {
        next();
        const std::string name = dotted_name();
        std::vector<ExprP> args;
        KwExprs kwargs;
        if (is_op("(")) parse_call_args(args, kwargs);
        if (!check_filter(name, args, kwargs, true)) {
            auto m = std::make_unique<Missing>();
            m->value = std::move(node);
            m->message = "No filter named '" + name + "' found.";
            return m;
        }
        auto f = std::make_unique<FilterExpr>();
        f->value = std::move(node);
        f->name = name;
        f->args = std::move(args);
        f->kwargs = std::move(kwargs);
        return f;
    }

    ExprP parse_test(ExprP node) {
        next();
        bool negated = false;
        if (is_name("not")) { next(); negated = true; }
        const std::string name = dotted_name();
        std::vector<ExprP> args;
        KwExprs kwargs;
        if (is_op("(")) parse_call_args(args, kwargs);
        else if ((cur().type == Token::NAME || cur().type == Token::STRING || cur().type == Token::INTEGER || cur().type == Token::FLOAT ||
                  is_op("(") || is_op("[") || is_op("{")) && !is_name("else") && !is_name("or") && !is_name("and")) {
            if (is_name("is")) refuse("tests cannot be chained with 'is'");
            args.push_back(parse_postfix(parse_primary()));
        }
        ExprP out;
        if (!check_filter(name, args, kwargs, false)) {
            auto m = std::make_unique<Missing>();
            m->value = std::move(node);
            m->message = "No test named '" + name + "' found.";
            out = std::move(m);
        } else {
            auto t = std::make_unique<TestExpr>();
            t->value = std::move(node);
            t->name = name;
            t->args = std::move(args);
            t->kwargs = std::move(kwargs);
            out = std::move(t);
        }
        return negated ? std::make_unique<Unary>('!', std::move(out)) : std::move(out);
    }
};

// A parsed template, rendered with the variables of a dict; one template serves concurrent renders.
class Template {
public:
    explicit Template(const std::string& source) {
        Parser parser(tokenize(source));
        body_ = parser.parse();
        fields_ = std::move(parser.fields);
        separators_ = std::move(parser.separators);
    }

    // Whether the template reads a field of this name, as an attribute, a constant subscript, a get() or a filter's attribute.
    bool reads(const std::string& field) const { return fields_.count(field) > 0; }
    // Whether it splits text at this constant separator.
    bool splits_at(const std::string& separator) const { return separators_.count(separator) > 0; }

    std::string render(const Value& context) const {
        if (context.k != Value::DICT) type_error("a template renders with a dict of variables");
        Render r{ context.obj.get() };
        Env env{ std::make_shared<Scope>(), &r };
        std::string out;
        exec_body(body_, env, out);
        return out;
    }

private:
    Body body_;
    std::set<std::string> fields_, separators_;
};

} // namespace jj

// The variables a chat render reads, as the reference renderer passes them for a conversation without tools or documents.
inline jj::Value context(const std::vector<Message>& messages, bool add_generation_prompt,
                         const std::string& bos_token, const std::string& eos_token) {
    using jj::Value;
    std::vector<Value> list;
    for (const auto& m : messages) {
        auto o = std::make_shared<jj::Object>();
        o->set("role", Value::str(m.role));
        o->set("content", Value::str(m.content));
        if (m.reasoning_content) o->set("reasoning_content", Value::str(*m.reasoning_content));
        list.push_back(Value::dict(o));
    }
    auto c = std::make_shared<jj::Object>();
    c->set("messages", Value::list(std::move(list)));
    c->set("tools", Value::none());
    c->set("documents", Value::none());
    c->set("add_generation_prompt", Value::boolean(add_generation_prompt));
    c->set("bos_token", Value::str(bos_token));
    c->set("eos_token", Value::str(eos_token));
    return Value::dict(c);
}

// How a model's conversations are written: its template parsed, or why it is refused, and the text of the start and end tokens a template may name.
struct ChatFormat {
    std::shared_ptr<const jj::Template> program;  // empty when the template is refused
    std::string refusal;
    std::string bos, eos;
    // Whether an assistant turn is kept split at </think>: the template reads `reasoning_content` and does not split a reply at </think> itself.
    bool split_turns = false;

    // An assistant turn as a conversation keeps it, the same for chat's own replies and the turns a client sends back.
    // It is split by assistant_turn only where the template needs the reasoning apart, and whole elsewhere: a template that splits a turn itself renders the whole text as the reference does, and one that knows no reasoning was written to take the turn as it came.
    Message assistant(const std::string& text) const {
        if (split_turns) return assistant_turn(text);
        return Message{ "assistant", text, std::nullopt };
    }

    // Raises the refusal; chat and serve call it before they take a turn.
    void require() const {
        if (!program) throw Refused("the model's chat template is refused: " + refusal);
    }
    // The prompt for `messages`, with the assistant's header after them when `add_generation_prompt`; a failing render raises TemplateError.
    std::string render(const std::vector<Message>& messages, bool add_generation_prompt) const {
        require();
        return program->render(context(messages, add_generation_prompt, bos, eos));
    }
};

// A template's source parsed once, or the reason it is refused.
inline ChatFormat chat_format(const std::string& source, std::string bos, std::string eos) {
    ChatFormat f;
    try { f.program = std::make_shared<const jj::Template>(source); }
    catch (const Refused& e) { f.refusal = e.what(); }
    f.split_turns = f.program && f.program->reads("reasoning_content") && !f.program->splits_at("</think>");
    f.bos = std::move(bos);
    f.eos = std::move(eos);
    return f;
}

// The template a file carries, or ChatML when it carries none, with its tokenizer's start and end text.
inline ChatFormat chat_format(const gguf::GGUFModel& m, const bpe::Tokenizer& tok) {
    const gguf::MetaValue* stored = m.find("tokenizer.chat_template");
    const bool own = stored && stored->vtype == gguf::V_STRING && !stored->s.empty();
    auto text = [&](int32_t id) { return id >= 0 && (size_t)id < tok.vocab.size() ? tok.vocab[(size_t)id] : std::string(); };
    return chat_format(own ? stored->s
                           : "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                             "{{ message['content'] }}<|im_end|>\n{% endfor %}"
                             "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
                       text(tok.bos_id), text(tok.eos_id));
}

} // namespace chat
