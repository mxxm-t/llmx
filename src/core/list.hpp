#pragma once
#include <string>
#include <vector>

namespace core {

// The entries of a comma-separated value, in order, empty ones included so the caller can refuse them.
inline std::vector<std::string> comma_list(const std::string& value) {
    std::vector<std::string> items;
    size_t start = 0;
    for (size_t comma; (comma = value.find(',', start)) != std::string::npos; start = comma + 1)
        items.push_back(value.substr(start, comma - start));
    items.push_back(value.substr(start));
    return items;
}

} // namespace core
