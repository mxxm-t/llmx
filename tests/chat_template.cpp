#include <fstream>
#include <iostream>
#include <iterator>
#include "core/json.hpp"
#include "inference/chat.hpp"

const jmini::Value& field(const jmini::Value& value, const char* name) {
    const auto* item = value.get(name);
    if (!item) throw std::runtime_error(std::string("missing fixture field: ") + name);
    return *item;
}

// Each expected string is what Jinja2 renders from the same messages when `tools` is passed as none, as it is for a request without tools.
struct DefinedCase { const char* tpl; const char* expected; };
const DefinedCase defined_cases[] = {
    // A date default of the kind other families' templates give a variable the caller did not pass.
    {R"({%- if not date_string is defined %}{%- set date_string = "26 Jul 2024" %}{%- endif %}Today Date: {{ date_string }})",
     "Today Date: 26 Jul 2024"},
    {R"({% if tools is defined and tools is none %}T{% endif %}{% if tool_calls is not defined and system is not defined %}N{% endif %}{% if messages is defined and eos_token is defined %}M{% endif %})",
     "TNM"},
    {R"({% for m in messages %}{{ m.role[0] }}{% if m.content is defined %}c{% endif %}{% if m.tool_calls is not defined %}-{% endif %}{% endfor %})",
     "sc-uc-"},
    {R"({% if messages[0]['content'] is defined %}K{% endif %}{% if messages[0]['name'] is not defined %}U{% endif %}{% if messages[1] is defined and messages[2] is not defined %}R{% endif %})",
     "KUR"},
    {R"({% set x = none %}{% set ns = namespace(a=none) %}{% if x is defined and ns.a is defined and ns.b is not defined %}D{% endif %})",
     "D"},
    {R"({% for m in messages %}{% if loop is defined %}L{% endif %}{% endfor %}{% if loop is not defined %}N{% endif %})",
     "LLN"},
};

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected chat-template fixture path");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open chat-template fixture");
        const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto fixture = jmini::Parser(text).parse();
        const auto& tpl = field(fixture, "template").str;
        size_t count = 0;
        for (const auto& test : field(fixture, "cases").arr) {
            std::vector<chat::Message> messages;
            for (const auto& message : field(test, "messages").arr)
                messages.push_back({field(message, "role").str, field(message, "content").str});
            const auto actual = chat::render(tpl, messages, field(test, "generate").b, "", "<|im_end|>");
            if (actual != field(test, "expected").str)
                throw std::runtime_error("Qwen template differs from Jinja2 at case " + std::to_string(count));
            ++count;
        }
        if (!count) throw std::runtime_error("empty chat-template fixture");
        std::cout << count << " Qwen chat-template cases match Jinja2\n";
        const std::vector<chat::Message> messages = {{"system", "S"}, {"user", "U"}};
        size_t differ = 0;
        for (size_t i = 0; i < std::size(defined_cases); ++i) {
            const auto actual = chat::render(defined_cases[i].tpl, messages, true, "", "<|im_end|>");
            if (actual == defined_cases[i].expected) continue;
            std::cerr << "defined case " << i << " renders \"" << actual << "\", Jinja2 renders \"" << defined_cases[i].expected << "\"\n";
            ++differ;
        }
        if (differ) throw std::runtime_error(std::to_string(differ) + " defined-test cases differ from Jinja2");
        std::cout << std::size(defined_cases) << " defined-test cases match Jinja2\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
