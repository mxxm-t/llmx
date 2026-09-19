// Include the real CLI emitter so removing its flush is observable without process-timing assumptions.
#define main llmx_cli_main
#include "../src/cli/main.cpp"
#undef main

class OutputBuffer : public std::stringbuf {
public:
    std::string delivered;
    int sync() override {
        delivered = str();
        return 0;
    }
};

int main() {
    OutputBuffer output;
    auto* saved = std::cout.rdbuf(&output);
    emit_text("A\xc3");
    const bool first = output.delivered == "A\xc3";
    emit_text("\xa9Z");
    const bool second = output.delivered == "A\xc3\xa9Z";
    std::cout.rdbuf(saved);
    if (!first || !second) {
        std::cerr << "CLI did not flush each byte chunk\n";
        return 1;
    }
    std::cout << "CLI output: each byte chunk flushed immediately\n";
    return 0;
}
