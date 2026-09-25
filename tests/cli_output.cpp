// Include the real CLI so its glue is tested as it runs: removing the emitter's flush is observable without process-timing assumptions, and --device lists are read as the commands read them.
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

// A --device list is canonical and each device appears once, however it is spelled; malformed entries are refused.
bool device_lists() {
    auto refused = [](const std::string& value) {
        try { backend::device_specs(value); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    return backend::device_specs("vulkan,cpu,vulkan:01") == std::vector<std::string>{"vulkan:0", "cpu", "vulkan:1"} &&
           refused("vulkan,vulkan:0") && refused("vulkan:00,vulkan:0") && refused("cpu,cpu") && refused("vulkan:0,") &&
           refused("vulkan:x") && refused("gpu:0") && layer_shares("3,1") == std::vector<int>{3, 1} && refused("");
}

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
    if (!device_lists()) {
        std::cerr << "CLI device lists not read canonically\n";
        return 1;
    }
    std::cout << "CLI output: each byte chunk flushed immediately; device lists canonical\n";
    return 0;
}
