#include "core/json.hpp"
#include <clocale>
#include <cmath>
#include <iostream>
#include <limits>
#include <locale>

size_t checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::string& input) {
    bool rejected = false;
    try { (void)jmini::parse(input); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid JSON accepted (" + std::to_string(input.size()) + " bytes)");
}

void number(const std::string& input, double expected) {
    const auto value = jmini::parse(input);
    require(value.isNumber() && value.asNumber() == expected,
            "wrong number: " + input);
    require(std::isfinite(value.asNumber()), "nonfinite number accepted");
    if (expected == 0) require(std::signbit(value.asNumber()) == std::signbit(expected), "zero sign changed");
}

void string_value(const std::string& input, const std::string& expected) {
    const auto value = jmini::parse(input);
    require(value.isString() && value.asString() == expected, "wrong decoded string bytes");
}

void values() {
    require(jmini::parse(" \t\r\nnull \n").t == jmini::Value::T::Null, "null/whitespace");
    const auto yes = jmini::parse("true"), no = jmini::parse("false");
    require(yes.t == jmini::Value::T::Bool && yes.b, "true");
    require(no.t == jmini::Value::T::Bool && !no.b, "false");
    const auto empty_array = jmini::parse("[]"), empty_object = jmini::parse("{}");
    require(empty_array.isArray() && empty_array.asArray().empty(), "empty array");
    require(empty_object.isObject() && empty_object.obj.empty(), "empty object");
    const auto value = jmini::parse("{\"a\":[null,false,1.25,\"x\",{}],\"a\":2,\"z\":true}");
    require(value.isObject() && value.obj.size() == 3, "object members");
    require(value.obj[0].first == "a" && value.obj[1].first == "a" && value.obj[2].first == "z", "member order/duplicates");
    require(value.get("a") == &value.obj[0].second && value.get("missing") == nullptr, "first duplicate lookup");
    const auto& array = value.get("a")->asArray();
    require(array.size() == 5 && array[0].t == jmini::Value::T::Null &&
            array[1].t == jmini::Value::T::Bool && !array[1].b &&
            array[2].asNumber() == 1.25 && array[3].asString() == "x" && array[4].isObject(), "nested values");
    require(array[2].get("a") == nullptr, "nonobject lookup");
    const auto key = jmini::parse("{\"a\\u0000b\":7,\"\\u00e9\":8}");
    const auto* nul_key = key.get(std::string("a\0b", 3));
    const auto* unicode_key = key.get("\xc3\xa9");
    require(nul_key && nul_key->asNumber() == 7, "NUL key");
    require(unicode_key && unicode_key->asNumber() == 8, "Unicode key");
}

void numbers() {
    for (const auto& entry : std::vector<std::pair<std::string, double>>{
             {"0", 0.0}, {"-0", -0.0}, {"-0.0e+10", -0.0}, {"0e999", 0.0},
             {"1", 1.0}, {"-12", -12.0}, {"12.375", 12.375}, {"1E2", 100.0},
             {"1e+2", 100.0}, {"125e-2", 1.25}, {"-1.25E+3", -1250.0},
             {"9007199254740992", 9007199254740992.0},
             {"1.7976931348623157e308", std::numeric_limits<double>::max()},
             {"2.2250738585072014e-308", std::numeric_limits<double>::min()},
             {"4.9406564584124654e-324", std::numeric_limits<double>::denorm_min()},
             {"-4.9406564584124654e-324", -std::numeric_limits<double>::denorm_min()},
             {"1e-308", 1e-308}, {"-1e-308", -1e-308},
             {"2.2250738585072009e-308", std::nextafter(std::numeric_limits<double>::min(), 0.0)},
             {"-2.2250738585072009e-308", -std::nextafter(std::numeric_limits<double>::min(), 0.0)},
             {"2.2250738585072012e-308", std::numeric_limits<double>::min()},
             {"-2.2250738585072012e-308", -std::numeric_limits<double>::min()},
             {"2.4703282292062328e-324", std::numeric_limits<double>::denorm_min()},
             {"-2.4703282292062328e-324", -std::numeric_limits<double>::denorm_min()}})
        number(entry.first, entry.second);
    for (const char* input : {"+1", ".5", "-.5", "1.", "01", "-01", "00", "0x10",
                              "1e", "1e+", "1e-", "1E--2", "1e+-2", "--1", "-", "1+2",
                              "1-2", "1.2.3", "1e2e3", "nan", "NaN", "Infinity", "-inf",
                              "2.4703282292062327e-324", "-2.4703282292062327e-324",
                              "1e309", "-1e309", "1e-324", "-1e-999", "1.7976931348623159e308"})
        rejects(input);
}

void strings() {
    string_value("\"\"", "");
    string_value("\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"", "\"\\/\b\f\n\r\t");
    string_value("\"a\\u0000b\"", std::string("a\0b", 3));
    for (const auto& entry : std::vector<std::pair<std::string, std::string>>{
             {"\\u007f", "\x7f"}, {"\\u0080", "\xc2\x80"},
             {"\\u07ff", "\xdf\xbf"}, {"\\u0800", "\xe0\xa0\x80"},
             {"\\uD7FF", "\xed\x9f\xbf"}, {"\\uE000", "\xee\x80\x80"},
             {"\\uffff", "\xef\xbf\xbf"}, {"\\u00e9", "\xc3\xa9"},
             {"\\u4e2d", "\xe4\xb8\xad"}, {"\\ud800\\udc00", "\xf0\x90\x80\x80"},
             {"\\uDBFF\\uDFFF", "\xf4\x8f\xbf\xbf"}, {"\\ud83d\\ude00", "\xf0\x9f\x98\x80"}}) {
        string_value('"' + entry.first + '"', entry.second);
        string_value('"' + entry.second + '"', entry.second);
    }
    for (unsigned c = 0; c < 32; ++c) rejects('"' + std::string(1, char(c)) + '"');
    for (const char* input : {"\"", "\"abc", "\"abc\\", "\"\\x\"", "\"\\v\"", "\"\\0\"",
                              "\"\\u\"", "\"\\u0\"", "\"\\u00\"", "\"\\u000\"", "\"\\u000g\"",
                              "\"\\u+123\"", "\"\\u 123\"", "\"\\ud800\"", "\"\\udfff\"",
                              "\"\\ud800x\"", "\"\\ud800\\u0041\"", "\"\\ud800\\ud800\"",
                              "\"\\udc00\\ud800\"", "\"\\ud800\\n\"", "\"\\ud800\\u12\""})
        rejects(input);
    for (const std::string& bytes : std::vector<std::string>{
             "\x80", "\xbf", "\xc0\x80", "\xc1\xbf", "\xc2", "\xdf", "\xc2 ",
             "\xe0\x80\x80", "\xe0\x9f\xbf", "\xe0\xa0", "\xe1\x80", "\xed\xa0\x80",
             "\xed\xbf\xbf", "\xef\xbf", "\xf0\x80\x80\x80", "\xf0\x8f\xbf\xbf",
             "\xf0\x90\x80", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80",
             "\xf8\x88\x80\x80\x80", "\xfe", "\xff", "\xe1\x41\x80", "\xf1\x80\x41\x80"})
        rejects('"' + bytes + '"');
}

void quoting() {
    for (const auto& entry : std::vector<std::pair<std::string, std::string>>{
             {"", "\"\""}, {"tensor.weight", "\"tensor.weight\""},
             {"a\"b\\c", "\"a\\\"b\\\\c\""},
             {std::string("a\0b\n", 4), "\"a\\u0000b\\u000a\""},
             {"\b\t\r\x1f", "\"\\u0008\\u0009\\u000d\\u001f\""},
             {"\xc3\xa9\xf0\x9f\x98\x80", "\"\xc3\xa9\xf0\x9f\x98\x80\""}}) {
        require(jmini::quote(entry.first) == entry.second, "wrong JSON quoting bytes");
        string_value(entry.second, entry.first);
    }
}

void structure() {
    for (const char* input : {"", " ", "[", "{", "[1", "{\"a\"", "{\"a\":", "{\"a\":1",
                              "[1,]", "{\"a\":1,}", "[,1]", "[1,,2]", "{,}", "[1 2]",
                              "{a:1}", "{\"a\" 1}", "{\"a\"::1}", "{\"a\":1 \"b\":2}",
                              "[}", "{]", "}", "]", "true false", "nullx", "false0", "1 true",
                              "{}[]", "\"x\"x", "t rue", "tr\tue", "fal se", "n\null", "True",
                              "tru", "fals", "nul", "/*x*/null", "[//x\n1]", "'x'", "\vnull", "null\f"})
        rejects(input);
    rejects(std::string("null\0", 5));
    rejects(std::string("\0null", 5));
    rejects("\xef\xbb\xbfnull");
    for (const char* valid : {"null", "true", "false", "0", "[]", "{}", "\"x\""}) {
        (void)jmini::parse(std::string(" \t") + valid + "\n\r");
        ++checks;
        rejects(std::string(valid) + "!");
    }
}

std::string nested(size_t depth, int kind) {
    std::string text = "0";
    for (size_t i = 0; i < depth; ++i)
        text = (kind == 0 || (kind == 2 && i % 2 == 0)) ? '[' + text + ']' : "{\"x\":" + text + '}';
    return text;
}

void nesting() {
    for (int kind : {0, 1, 2}) {
        const auto value = jmini::parse(nested(256, kind));
        const jmini::Value* leaf = &value;
        size_t depth = 0;
        while (leaf->isArray() || leaf->isObject()) {
            ++depth;
            if (leaf->isArray()) {
                require(leaf->arr.size() == 1, "nested array extent");
                leaf = &leaf->arr[0];
            } else {
                require(leaf->obj.size() == 1 && leaf->get("x"), "nested object extent");
                leaf = leaf->get("x");
            }
        }
        require(depth == 256 && leaf->isNumber() && leaf->asNumber() == 0, "maximum nesting");
        rejects(nested(257, kind));
    }
    require(jmini::parse('[' + std::string(255, '[') + std::string(255, ']') + ']').isArray(), "empty deepest container");
    require(jmini::parse('[' + nested(255, 0) + ',' + nested(255, 1) + ']').arr.size() == 2, "sibling depth reset");
}

struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

struct RestoreLocale {
    std::locale cpp = std::locale();
    std::string c_all = std::setlocale(LC_ALL, nullptr);
    ~RestoreLocale() {
        std::locale::global(cpp);
        std::setlocale(LC_ALL, c_all.c_str());
    }
};

void locales() {
    const std::locale before_cpp;
    const std::string before_c = std::setlocale(LC_ALL, nullptr);
    bool comma_c = false;
    {
        RestoreLocale restore;
        std::locale::global(std::locale(restore.cpp, new CommaDecimal));
        for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "German_Germany.1252", "German"}) {
            if (std::setlocale(LC_NUMERIC, name) && std::localeconv()->decimal_point[0] == ',') {
                comma_c = true;
                break;
            }
        }
        number("1.25", 1.25);
        number("12e-1", 1.2);
        rejects("1,25");
        rejects("1.234,5");
    }
    require(std::locale() == before_cpp && std::setlocale(LC_ALL, nullptr) == before_c, "locale not restored");
    std::cout << "JSON locale: custom C++ comma facet; installed C comma locale "
              << (comma_c ? "checked" : "unavailable") << '\n';
}

int main() {
    try {
        values();
        numbers();
        strings();
        quoting();
        structure();
        nesting();
        locales();
        std::cout << "JSON: " << checks << " independent checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
