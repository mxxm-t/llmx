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
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
