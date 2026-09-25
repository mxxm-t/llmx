#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include "tokenizer/tokenizer.hpp"
#include "inference/chat.hpp"
#include "core/json.hpp"
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: chat-prompt MODEL TEXT");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open text");
        const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto model = gguf::read_gguf(argv[1]);
        const bpe::Tokenizer tok(model);
        const std::string tpl = chat::get_chat_template(model);
        if (tpl.empty()) throw std::runtime_error("missing model chat template");
        const std::string bos = tok.bos_id >= 0 && (size_t)tok.bos_id < tok.vocab.size() ? tok.vocab[tok.bos_id] : "";
        const std::string eos = tok.eos_id >= 0 && (size_t)tok.eos_id < tok.vocab.size() ? tok.vocab[tok.eos_id] : "";
        const std::string prompt = chat::render(tpl, {{"user", text}}, true, bos, eos);
        std::cout << "{\"template\":" << jmini::quote(tpl) << ",\"bos\":" << jmini::quote(bos)
                  << ",\"eos\":" << jmini::quote(eos) << ",\"prompt\":" << jmini::quote(prompt) << ",\"ids\":[";
        bool first = true;
        for (uint32_t id : tok.encode(prompt)) { if (!first) std::cout << ','; std::cout << id; first = false; }
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
