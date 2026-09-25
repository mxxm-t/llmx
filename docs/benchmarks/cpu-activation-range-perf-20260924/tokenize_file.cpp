#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: tokenize-file MODEL TEXT");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open text");
        const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto model = gguf::read_gguf(argv[1]);
        const bpe::Tokenizer tokenizer(model);
        for (uint32_t id : tokenizer.encode(text)) std::cout << id << ' ';
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
