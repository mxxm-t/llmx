#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cctype>
#include <cstdlib>

#include "gguf.hpp"

// Minimal Jinja2-subset renderer for GGUF `tokenizer.chat_template` strings.
// Supports the control-flow and expressions used by common chat templates
// (Qwen2/3, Llama, Mistral, Gemma): {{ ... }} output, {% if/elif/else/for/set %},
// dict/list/string access, .get()/.keys()/etc., and the `messages`,
// `add_generation_prompt`, `bos_token`, `eos_token` context variables.
// Undefined variables evaluate to none (empty) so unknown templates degrade
// gracefully rather than throwing.

namespace chat {

struct Message {
    std::string role;
    std::string content;
};

namespace jj {

struct Value {
    enum T { NONE, BOOL, NUM, STR, LIST, DICT } t = NONE;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Value> list;
    std::map<std::string, Value> dict;

    static Value none() { return Value(); }
    static Value boolean(bool v) { Value x; x.t = BOOL; x.b = v; return x; }
    static Value number(double v) { Value x; x.t = NUM; x.n = v; return x; }
    static Value str(const std::string& v) { Value x; x.t = STR; x.s = v; return x; }
    static Value arr() { Value x; x.t = LIST; return x; }
    static Value obj() { Value x; x.t = DICT; return x; }

    bool truthy() const {
        switch (t) {
            case NONE: return false;
            case BOOL: return b;
            case NUM:  return n != 0;
            case STR:  return !s.empty();
            case LIST: return !list.empty();
            case DICT: return !dict.empty();
        }
        return false;
    }

    std::string to_str() const {
        switch (t) {
            case STR: return s;
            case NUM: {
                if (n == (long long)n) return std::to_string((long long)n);
                std::string r = std::to_string(n);
                while (!r.empty() && r.back() == '0') r.pop_back();
                if (!r.empty() && r.back() == '.') r.pop_back();
                return r;
            }
            case BOOL: return b ? "true" : "false";
            case NONE: return "";
            case LIST: {
                std::string o;
                for (auto& e : list) { if (!o.empty()) o += ", "; o += e.to_str(); }
                return "[" + o + "]";
            }
            case DICT: return "";
        }
        return "";
    }

    bool val_eq(const Value& o) const {
        if (t != o.t) {
            // numeric compare across types
            if ((t == NUM || t == BOOL) && (o.t == NUM || o.t == BOOL))
                return (t == NUM ? n : (b ? 1 : 0)) == (o.t == NUM ? o.n : (o.b ? 1 : 0));
            return false;
        }
        switch (t) {
            case NONE: return true;
            case BOOL: return b == o.b;
            case NUM:  return n == o.n;
            case STR:  return s == o.s;
            case LIST: {
                if (list.size() != o.list.size()) return false;
                for (size_t i = 0; i < list.size(); i++)
                    if (!list[i].val_eq(o.list[i])) return false;
                return true;
            }
            case DICT: {
                if (dict.size() != o.dict.size()) return false;
                for (auto& kv : dict) {
                    auto it = o.dict.find(kv.first);
                    if (it == o.dict.end() || !kv.second.val_eq(it->second)) return false;
                }
                return true;
            }
        }
        return false;
    }
};

struct Ctx { std::map<std::string, Value> vars; };

struct Expr {
    virtual ~Expr() {}
    virtual Value eval(const Ctx&) const = 0;
};

struct Lit : Expr { Value v; explicit Lit(const Value& x) : v(x) {} Value eval(const Ctx&) const { return v; } };
struct Var : Expr { std::string name; explicit Var(const std::string& n) : name(n) {}
    Value eval(const Ctx& c) const {
        auto it = c.vars.find(name);
        return (it == c.vars.end()) ? Value::none() : it->second;
    }
};
struct Attr : Expr { std::shared_ptr<Expr> base; std::string key;
    Attr(const std::shared_ptr<Expr>& b, const std::string& k) : base(b), key(k) {}
    Value eval(const Ctx& c) const {
        Value v = base->eval(c);
        if (v.t == Value::DICT) {
            auto it = v.dict.find(key);
            if (it != v.dict.end()) return it->second;
        }
        return Value::none();
    }
};
struct Index : Expr { std::shared_ptr<Expr> base, idx;
    Index(const std::shared_ptr<Expr>& b, const std::shared_ptr<Expr>& i) : base(b), idx(i) {}
    Value eval(const Ctx& c) const {
        Value v = base->eval(c);
        Value i = idx->eval(c);
        if (v.t == Value::LIST && i.t == Value::NUM) {
            long k = (long)i.n;
            if (k < 0) k += (long)v.list.size();
            if (k >= 0 && (size_t)k < v.list.size()) return v.list[(size_t)k];
            return Value::none();
        }
        if (v.t == Value::DICT && i.t == Value::STR) {
            auto it = v.dict.find(i.s);
            if (it != v.dict.end()) return it->second;
        }
        if (v.t == Value::STR && i.t == Value::NUM) {
            long k = (long)i.n;
            if (k >= 0 && (size_t)k < v.s.size()) return Value::str(std::string(1, v.s[(size_t)k]));
        }
        return Value::none();
    }
};
struct Unary : Expr { std::string op; std::shared_ptr<Expr> e;
    Unary(const std::string& o, const std::shared_ptr<Expr>& x) : op(o), e(x) {}
    Value eval(const Ctx& c) const {
        Value v = e->eval(c);
        if (op == "not") return Value::boolean(!v.truthy());
        if (op == "-") return Value::number(-v.n);
        return Value::none();
    }
};
struct Binary : Expr { std::string op; std::shared_ptr<Expr> l, r;
    Binary(const std::string& o, const std::shared_ptr<Expr>& a, const std::shared_ptr<Expr>& b) : op(o), l(a), r(b) {}
    Value eval(const Ctx& c) const {
        Value a = l->eval(c), b = r->eval(c);
        if (op == "or") return Value::boolean(a.truthy() || b.truthy());
        if (op == "and") return Value::boolean(a.truthy() && b.truthy());
        if (op == "==") return Value::boolean(a.val_eq(b));
        if (op == "!=") return Value::boolean(!a.val_eq(b));
        if (op == "in")  return Value::boolean(contains(a, b));
        if (op == "not in") return Value::boolean(!contains(a, b));
        if (op == "~") return Value::str(a.to_str() + b.to_str());
        if (op == "+") {
            if (a.t == Value::NUM && b.t == Value::NUM) return Value::number(a.n + b.n);
            if (a.t == Value::LIST && b.t == Value::LIST) { Value v = a; v.list.insert(v.list.end(), b.list.begin(), b.list.end()); return v; }
            return Value::str(a.to_str() + b.to_str());
        }
        if (op == "-") return Value::number(a.n - b.n);
        if (op == "*") return Value::number(a.n * b.n);
        if (op == "/") return b.n == 0 ? Value::number(0) : Value::number(a.n / b.n);
        if (op == "//") return b.n == 0 ? Value::number(0) : Value::number(std::floor(a.n / b.n));
        if (op == "%") return b.n == 0 ? Value::number(0) : Value::number(std::fmod(a.n, b.n));
        if (op == "<") return Value::boolean(a.t == Value::STR ? (a.s < b.s) : (a.n < b.n));
        if (op == ">") return Value::boolean(a.t == Value::STR ? (a.s > b.s) : (a.n > b.n));
        if (op == "<=") return Value::boolean(a.t == Value::STR ? (a.s <= b.s) : (a.n <= b.n));
        if (op == ">=") return Value::boolean(a.t == Value::STR ? (a.s >= b.s) : (a.n >= b.n));
        return Value::none();
    }
    static bool contains(const Value& item, const Value& container) {
        if (container.t == Value::LIST) {
            for (auto& e : container.list) if (e.val_eq(item)) return true;
            return false;
        }
        if (container.t == Value::DICT) return container.dict.count(item.to_str()) > 0;
        if (container.t == Value::STR) return container.s.find(item.to_str()) != std::string::npos;
        return false;
    }
};
struct Call : Expr { std::shared_ptr<Expr> callee; std::vector<std::shared_ptr<Expr>> args;
    Call(const std::shared_ptr<Expr>& c, const std::vector<std::shared_ptr<Expr>>& a) : callee(c), args(a) {}
    Value eval(const Ctx& c) const {
        std::vector<Value> av;
        av.reserve(args.size());
        for (auto& a : args) av.push_back(a->eval(c));
        if (auto v = dynamic_cast<Var*>(callee.get())) return call_global(v->name, av);
        if (auto at = dynamic_cast<Attr*>(callee.get())) {
            Value base = at->base->eval(c);
            return call_method(at->key, base, av);
        }
        return Value::none();
    }
    static Value call_method(const std::string& name, const Value& base, const std::vector<Value>& args) {
        if (name == "get") {
            std::string key = args.empty() ? "" : args[0].to_str();
            if (base.t == Value::DICT) {
                auto it = base.dict.find(key);
                if (it != base.dict.end()) return it->second;
            }
            return args.size() > 1 ? args[1] : Value::none();
        }
        if (name == "keys") {
            Value r = Value::arr();
            for (auto& kv : base.dict) r.list.push_back(Value::str(kv.first));
            return r;
        }
        if (name == "items") {
            Value r = Value::arr();
            for (auto& kv : base.dict) {
                Value p = Value::arr();
                p.list.push_back(Value::str(kv.first));
                p.list.push_back(kv.second);
                r.list.push_back(p);
            }
            return r;
        }
        std::string s = base.to_str();
        if (name == "strip") {
            size_t a = 0; while (a < s.size() && std::isspace((unsigned char)s[a])) a++;
            size_t b = s.size(); while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
            return Value::str(s.substr(a, b - a));
        }
        if (name == "lower") { for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch); return Value::str(s); }
        if (name == "upper") { for (auto& ch : s) ch = (char)std::toupper((unsigned char)ch); return Value::str(s); }
        if (name == "title") {
            bool up = true;
            for (auto& ch : s) { if (std::isalpha((unsigned char)ch)) { ch = up ? (char)std::toupper((unsigned char)ch) : (char)std::tolower((unsigned char)ch); up = false; } else up = true; }
            return Value::str(s);
        }
        if (name == "replace" && args.size() >= 2) {
            const std::string& f = args[0].to_str();
            const std::string& r = args[1].to_str();
            size_t pos = 0; std::string out;
            while ((pos = s.find(f, pos)) != std::string::npos) { out += s.substr(0, pos) + r; s.erase(0, pos + f.size()); pos = 0; }
            out += s;
            return Value::str(out);
        }
        if (name == "count" && !args.empty()) {
            const std::string& f = args[0].to_str();
            size_t pos = 0, cnt = 0;
            while ((pos = s.find(f, pos)) != std::string::npos) { cnt++; pos += f.size(); }
            return Value::number((double)cnt);
        }
        if (name == "split" && !args.empty()) {
            Value r = Value::arr();
            const std::string& sep = args[0].to_str();
            size_t pos = 0;
            while (true) {
                size_t n = s.find(sep, pos);
                if (n == std::string::npos) { r.list.push_back(Value::str(s.substr(pos))); break; }
                r.list.push_back(Value::str(s.substr(pos, n - pos)));
                pos = n + sep.size();
            }
            return r;
        }
        if (name == "length" || name == "len") {
            if (base.t == Value::LIST) return Value::number((double)base.list.size());
            if (base.t == Value::DICT) return Value::number((double)base.dict.size());
            return Value::number((double)s.size());
        }
        return Value::none();
    }
    static Value call_global(const std::string& name, const std::vector<Value>& args) {
        if (name == "range") {
            long n = args.empty() ? 0 : (long)args[0].n;
            Value r = Value::arr();
            for (long i = 0; i < n; i++) r.list.push_back(Value::number((double)i));
            return r;
        }
        if (name == "namespace") {
            Value o = Value::obj();
            for (size_t i = 0; i + 1 < args.size(); i += 2) o.dict[args[i].to_str()] = args[i + 1];
            return o;
        }
        return Value::none();
    }
};

// ---- expression lexer ----
struct Tok {
    enum K { ID, NUM, STR, OP, LP, RP, LB, RB, COMMA, DOT, END } k;
    std::string text;
    double num = 0;
};

inline std::vector<Tok> lex_expr(const std::string& s) {
    std::vector<Tok> ts;
    size_t i = 0, n = s.size();
    while (i < n) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { i++; continue; }
        if (std::isalpha((unsigned char)c) || c == '_') {
            size_t j = i;
            while (j < n && (std::isalnum((unsigned char)s[j]) || s[j] == '_')) j++;
            ts.push_back({ Tok::ID, s.substr(i, j - i), 0 }); i = j;
        } else if (std::isdigit((unsigned char)c) || (c == '.' && i + 1 < n && std::isdigit((unsigned char)s[i + 1]))) {
            size_t j = i;
            while (j < n && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) j++;
            ts.push_back({ Tok::NUM, "", std::strtod(s.substr(i, j - i).c_str(), nullptr) }); i = j;
        } else if (c == '\'' || c == '"') {
            char q = c; i++;
            std::string v;
            while (i < n && s[i] != q) {
                if (s[i] == '\\' && i + 1 < n) {
                    char e = s[i + 1];
                    switch (e) {
                        case 'n': v += '\n'; break;
                        case 't': v += '\t'; break;
                        case 'r': v += '\r'; break;
                        case '\\': v += '\\'; break;
                        case '\'': v += '\''; break;
                        case '"': v += '"'; break;
                        default: v += e;
                    }
                    i += 2;
                } else { v += s[i]; i++; }
            }
            i++;
            ts.push_back({ Tok::STR, v, 0 });
        } else {
            std::string two = s.substr(i, 2);
            if (two == "==" || two == "!=" || two == "<=" || two == ">=" || two == "//") { ts.push_back({ Tok::OP, two, 0 }); i += 2; }
            else if (c == '(') { ts.push_back({ Tok::LP, "(", 0 }); i++; }
            else if (c == ')') { ts.push_back({ Tok::RP, ")", 0 }); i++; }
            else if (c == '[') { ts.push_back({ Tok::LB, "[", 0 }); i++; }
            else if (c == ']') { ts.push_back({ Tok::RB, "]", 0 }); i++; }
            else if (c == ',') { ts.push_back({ Tok::COMMA, ",", 0 }); i++; }
            else if (c == '.') { ts.push_back({ Tok::DOT, ".", 0 }); i++; }
            else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '~' || c == '<' || c == '>' || c == '=') { ts.push_back({ Tok::OP, std::string(1, c), 0 }); i++; }
            else i++;
        }
    }
    ts.push_back({ Tok::END, "", 0 });
    return ts;
}

struct ExprParser {
    std::vector<Tok> ts;
    size_t p = 0;
    const Tok& cur() const { return ts[p]; }
    bool is_op(const char* s) const { return ts[p].k == Tok::OP && ts[p].text == s; }
    bool is_id(const char* s) const { return ts[p].k == Tok::ID && ts[p].text == s; }
    void adv() { if (p + 1 < ts.size()) p++; }

    std::shared_ptr<Expr> parse() { return parse_or(); }
    std::shared_ptr<Expr> parse_or() {
        auto l = parse_and();
        while (is_id("or")) { adv(); auto r = parse_and(); l = std::make_shared<Binary>("or", l, r); }
        return l;
    }
    std::shared_ptr<Expr> parse_and() {
        auto l = parse_not();
        while (is_id("and")) { adv(); auto r = parse_not(); l = std::make_shared<Binary>("and", l, r); }
        return l;
    }
    std::shared_ptr<Expr> parse_not() {
        if (is_id("not")) { adv(); return std::make_shared<Unary>("not", parse_not()); }
        return parse_comp();
    }
    std::shared_ptr<Expr> parse_comp() {
        auto l = parse_add();
        if (is_op("==") || is_op("!=") || is_op("<") || is_op(">") || is_op("<=") || is_op(">=")) {
            std::string op = cur().text; adv();
            auto r = parse_add();
            return std::make_shared<Binary>(op, l, r);
        }
        if (is_id("in")) { adv(); auto r = parse_add(); return std::make_shared<Binary>("in", l, r); }
        if (is_id("not") && p + 1 < ts.size() && ts[p + 1].k == Tok::ID && ts[p + 1].text == "in") {
            adv(); adv(); auto r = parse_add(); return std::make_shared<Binary>("not in", l, r);
        }
        return l;
    }
    std::shared_ptr<Expr> parse_add() {
        auto l = parse_mul();
        while (is_op("+") || is_op("-") || is_op("~")) {
            std::string op = cur().text; adv();
            auto r = parse_mul();
            l = std::make_shared<Binary>(op, l, r);
        }
        return l;
    }
    std::shared_ptr<Expr> parse_mul() {
        auto l = parse_unary();
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            std::string op = cur().text; adv();
            auto r = parse_unary();
            l = std::make_shared<Binary>(op, l, r);
        }
        return l;
    }
    std::shared_ptr<Expr> parse_unary() {
        if (is_op("-")) { adv(); return std::make_shared<Unary>("-", parse_unary()); }
        return parse_postfix();
    }
    std::shared_ptr<Expr> parse_postfix() {
        auto e = parse_primary();
        while (true) {
            if (cur().k == Tok::DOT) {
                adv();
                if (cur().k == Tok::ID) { std::string k = cur().text; adv(); e = std::make_shared<Attr>(e, k); }
                else break;
            } else if (cur().k == Tok::LP) {
                adv();
                std::vector<std::shared_ptr<Expr>> args;
                if (cur().k != Tok::RP) {
                    args.push_back(parse());
                    while (cur().k == Tok::COMMA) { adv(); args.push_back(parse()); }
                }
                adv(); // ')'
                e = std::make_shared<Call>(e, args);
            } else if (cur().k == Tok::LB) {
                adv();
                auto idx = parse();
                if (cur().k == Tok::RB) adv();
                e = std::make_shared<Index>(e, idx);
            } else break;
        }
        return e;
    }
    std::shared_ptr<Expr> parse_primary() {
        if (cur().k == Tok::NUM) { Value v = Value::number(cur().num); adv(); return std::make_shared<Lit>(v); }
        if (cur().k == Tok::STR) { Value v = Value::str(cur().text); adv(); return std::make_shared<Lit>(v); }
        if (is_id("true")) { adv(); return std::make_shared<Lit>(Value::boolean(true)); }
        if (is_id("false")) { adv(); return std::make_shared<Lit>(Value::boolean(false)); }
        if (is_id("none") || is_id("null") || is_id("True") || is_id("False") || is_id("None")) {
            std::string t = cur().text; adv();
            if (t == "True" || t == "true") return std::make_shared<Lit>(Value::boolean(true));
            if (t == "False" || t == "false") return std::make_shared<Lit>(Value::boolean(false));
            return std::make_shared<Lit>(Value::none());
        }
        if (cur().k == Tok::ID) { std::string name = cur().text; adv(); return std::make_shared<Var>(name); }
        if (cur().k == Tok::LP) { adv(); auto e = parse(); if (cur().k == Tok::RP) adv(); return e; }
        return std::make_shared<Lit>(Value::none());
    }
};

inline std::shared_ptr<Expr> parse_expr(const std::string& src) {
    ExprParser p;
    p.ts = lex_expr(src);
    p.p = 0;
    return p.parse();
}

// ---- template nodes ----
struct Node {
    virtual ~Node() {}
    virtual std::string render(Ctx&) const = 0;
};
struct Text : Node { std::string s; explicit Text(const std::string& x) : s(x) {} std::string render(Ctx&) const { return s; } };
struct Output : Node { std::shared_ptr<Expr> e; explicit Output(const std::shared_ptr<Expr>& x) : e(x) {} std::string render(Ctx& c) const { return e->eval(c).to_str(); } };
struct IfNode : Node {
    std::vector<std::pair<std::shared_ptr<Expr>, std::vector<std::shared_ptr<Node>>>> branches;
    std::vector<std::shared_ptr<Node>> elseBody;
    std::string render(Ctx& c) const {
        for (auto& br : branches) if (br.first->eval(c).truthy()) { std::string o; for (auto& n : br.second) o += n->render(c); return o; }
        std::string o; for (auto& n : elseBody) o += n->render(c); return o;
    }
};
struct ForNode : Node {
    std::string var;
    std::shared_ptr<Expr> iter;
    std::vector<std::shared_ptr<Node>> body;
    std::string render(Ctx& c) const {
        Value it = iter->eval(c);
        std::string out;
        if (it.t == Value::LIST) {
            size_t n = it.list.size();
            for (size_t i = 0; i < n; i++) {
                c.vars[var] = it.list[i];
                Value loop = Value::obj();
                loop.dict["index"] = Value::number((double)i + 1);
                loop.dict["index0"] = Value::number((double)i);
                loop.dict["first"] = Value::boolean(i == 0);
                loop.dict["last"] = Value::boolean(i + 1 == n);
                c.vars["loop"] = loop;
                for (auto& nd : body) out += nd->render(c);
            }
        } else if (it.t == Value::DICT) {
            for (auto& kv : it.dict) { c.vars[var] = Value::str(kv.first); for (auto& nd : body) out += nd->render(c); }
        }
        return out;
    }
};
struct SetNode : Node { std::string name; std::shared_ptr<Expr> e; std::string render(Ctx& c) const { c.vars[name] = e->eval(c); return ""; } };

inline std::string strip_ws(const std::string& s) {
    std::string t = s;
    size_t a = 0;
    while (a < t.size() && std::isspace((unsigned char)t[a])) a++;
    t = t.substr(a);
    if (!t.empty() && (t[0] == '-' || t[0] == '+')) t = t.substr(1);
    a = 0;
    while (a < t.size() && std::isspace((unsigned char)t[a])) a++;
    t = t.substr(a);
    size_t b = t.size();
    while (b > 0 && std::isspace((unsigned char)t[b - 1])) b--;
    t = t.substr(0, b);
    if (!t.empty() && (t.back() == '-' || t.back() == '+')) {
        t.pop_back();
        b = t.size();
        while (b > 0 && std::isspace((unsigned char)t[b - 1])) b--;
        t = t.substr(0, b);
    }
    return t;
}

inline std::vector<std::shared_ptr<Node>> parse_nodes(
    const std::vector<std::pair<std::string, std::string>>& blk, size_t& i,
    std::string& terminator) {
    std::vector<std::shared_ptr<Node>> nodes;
    while (i < blk.size()) {
        const auto& [kind, raw] = blk[i];
        if (kind == "text") {
            nodes.push_back(std::make_shared<Text>(raw)); i++;
        } else if (kind == "output") {
            nodes.push_back(std::make_shared<Output>(parse_expr(strip_ws(raw)))); i++;
        } else if (kind == "comment") {
            i++;
        } else if (kind == "tag") {
            std::string t = strip_ws(raw);
            size_t sp = t.find(' ');
            std::string name = (sp == std::string::npos) ? t : t.substr(0, sp);
            std::string rest = (sp == std::string::npos) ? "" : t.substr(sp + 1);

            if (name == "if") {
                auto node = std::make_shared<IfNode>();
                auto cond = parse_expr(rest);
                i++;
                std::string term;
                auto body = parse_nodes(blk, i, term);
                node->branches.push_back({ cond, body });
                while (term == "elif") {
                    // blk[i] is the {% elif ... %} tag; drop the leading keyword.
                    std::string eraw = strip_ws(blk[i].second);
                    size_t esp = eraw.find(' ');
                    std::string econd = (esp == std::string::npos) ? "" : eraw.substr(esp + 1);
                    auto cond2 = parse_expr(econd);
                    i++;
                    std::string t2;
                    auto body2 = parse_nodes(blk, i, t2);
                    node->branches.push_back({ cond2, body2 });
                    term = t2;
                }
                if (term == "else") {
                    i++;
                    std::string t3;
                    node->elseBody = parse_nodes(blk, i, t3);
                    term = t3;
                }
                if (term != "endif") throw std::runtime_error("chat template: unbalanced {% if %}");
                i++;
                nodes.push_back(node);
            } else if (name == "for") {
                // rest: "var in expr"
                size_t inpos = rest.find(" in ");
                if (inpos == std::string::npos) throw std::runtime_error("chat template: bad {% for %}");
                std::string var = rest.substr(0, inpos);
                std::string iter = rest.substr(inpos + 4);
                auto node = std::make_shared<ForNode>();
                node->var = var;
                node->iter = parse_expr(iter);
                i++;
                std::string term;
                node->body = parse_nodes(blk, i, term);
                if (term != "endfor") throw std::runtime_error("chat template: unbalanced {% for %}");
                i++;
                nodes.push_back(node);
            } else if (name == "set") {
                size_t eq = rest.find('=');
                if (eq == std::string::npos) throw std::runtime_error("chat template: bad {% set %}");
                auto node = std::make_shared<SetNode>();
                node->name = rest.substr(0, eq);
                // trim
                size_t a = 0; while (a < node->name.size() && std::isspace((unsigned char)node->name[a])) a++;
                size_t b = node->name.size(); while (b > a && std::isspace((unsigned char)node->name[b - 1])) b--;
                node->name = node->name.substr(a, b - a);
                node->e = parse_expr(rest.substr(eq + 1));
                i++;
                nodes.push_back(node);
            } else if (name == "endif" || name == "endfor" || name == "else" || name == "elif") {
                terminator = name;
                i++;
                return nodes;
            } else {
                i++;
            }
        }
    }
    terminator = "";
    return nodes;
}

inline std::vector<std::pair<std::string, std::string>> lex_template(const std::string& s) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t best = n;
        std::string marker;
        for (const char* mk : { "{{", "{%", "{#" }) {
            size_t pos = s.find(mk, i);
            if (pos != std::string::npos && pos < best) { best = pos; marker = mk; }
        }
        if (marker.empty()) { out.emplace_back("text", s.substr(i)); break; }
        if (best > i) out.emplace_back("text", s.substr(i, best - i));
        std::string close = (marker == "{{") ? "}}" : (marker == "{%") ? "%}" : "#}";
        size_t epos = s.find(close, best + 2);
        if (epos == std::string::npos) { out.emplace_back("text", s.substr(best)); break; }
        std::string content = s.substr(best + 2, epos - (best + 2));
        std::string kind = (marker == "{{") ? "output" : (marker == "{%") ? "tag" : "comment";
        out.emplace_back(kind, content);
        i = epos + 2;
    }
    return out;
}

} // namespace jj

// Render a chat template with the given message list.
// `add_generation_prompt` appends the trailing assistant header if the template
// opts in via `{% if add_generation_prompt %}`.
inline std::string render(const std::string& tpl,
                          const std::vector<Message>& messages,
                          bool add_generation_prompt,
                          const std::string& bos_token,
                          const std::string& eos_token) {
    auto blk = jj::lex_template(tpl);
    size_t i = 0;
    std::string term;
    auto nodes = jj::parse_nodes(blk, i, term);

    jj::Ctx ctx;
    jj::Value ml = jj::Value::arr();
    for (const auto& m : messages) {
        jj::Value d = jj::Value::obj();
        d.dict["role"] = jj::Value::str(m.role);
        d.dict["content"] = jj::Value::str(m.content);
        ml.list.push_back(std::move(d));
    }
    ctx.vars["messages"] = ml;
    ctx.vars["add_generation_prompt"] = jj::Value::boolean(add_generation_prompt);
    ctx.vars["bos_token"] = jj::Value::str(bos_token);
    ctx.vars["eos_token"] = jj::Value::str(eos_token);
    ctx.vars["tools"] = jj::Value::none();
    ctx.vars["tool_calls"] = jj::Value::none();
    ctx.vars["system"] = jj::Value::none();

    std::string out;
    for (auto& nd : nodes) out += nd->render(ctx);
    return out;
}

// Read the chat template metadata key from a GGUF model. Returns empty if
// absent.
inline std::string get_chat_template(const gguf::GGUFModel& m) {
    for (const auto& kv : m.kv) {
        if (kv.first == "tokenizer.chat_template" && kv.second.vtype == gguf::V_STRING)
            return kv.second.s;
    }
    return "";
}

} // namespace chat
