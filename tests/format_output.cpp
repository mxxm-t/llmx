#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include "format/file_writer.hpp"
#include "format/output_file.hpp"

namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::string read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot read test output");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void put(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << text;
    out.close();
}

// The caller supplies a new directory in its build tree; cleanup owns only the directory it creates.
struct Directory {
    std::filesystem::path path;
    explicit Directory(const char* name) : path(name) {
        require(std::filesystem::create_directory(path), "test directory already exists");
    }
    ~Directory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void check(const std::filesystem::path& dir) {
    const auto first = dir / "first", second = dir / "second";
    put(first, "old");
    {
        format::OutputFile output(first.u8string());
        bool caught = false;
        try {
            output.write([](std::ostream& os) {
                os << "partial";
                throw std::runtime_error("serializer failed");
            });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "serializer failed";
        }
        require(caught, "serialization failure was hidden");
        require(read(first) == "old", "serialization replaced destination");
    }
    {
        format::OutputFile output(first.u8string());
        bool caught = false;
        try {
            output.write([](std::ostream& os) { os.setstate(std::ios::badbit); });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()).find(first.u8string()) != std::string::npos;
        }
        require(caught, "stream failure did not name destination");
    }
    {
        format::OutputFile output(first.u8string());
        bool caught = false;
        try { output.publish(); }
        catch (const std::logic_error&) { caught = true; }
        require(caught && read(first) == "old", "uncompleted output was published");
    }
    {
        format::OutputFile a(first.u8string()), b(second.u8string());
        a.write([](std::ostream& os) { os << "complete first"; });
        b.write([](std::ostream& os) { os << "complete second"; });
        require(read(first) == "old" && !std::filesystem::exists(second), "staging published outputs");
        // Deterministically refuse the second publication after both writes/close and the first rename.
        require(std::filesystem::create_directory(second), "cannot block second publication");
        put(second / "keep", "untouched");
        a.publish();
        bool caught = false;
        try { b.publish(); }
        catch (const std::runtime_error& e) {
            caught = std::string(e.what()).find(second.u8string()) != std::string::npos;
        }
        require(caught, "publication failure did not name destination");
        require(read(first) == "complete first", "first publication is not complete");
        require(read(second / "keep") == "untouched", "failed publication changed destination");
    }
    {
        format::OutputFile text(first.u8string(), std::ios::out);
        text.write([](std::ostream& os) { os << "line\n"; });
        text.publish();
#if defined(_WIN32)
        require(read(first) == "line\r\n", "text output changed Windows line endings");
#else
        require(read(first) == "line\n", "text output changed line endings");
#endif
    }
    size_t entries = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        require(entry.path() == first || entry.path() == second, "temporary output left behind");
        ++entries;
    }
    require(entries == 2, "destination removed during cleanup");
}

// A symbolic link at a writer's path, dangling here, must fail the open as an existing file does and never create the link's target.
// On Windows a link needs developer mode or a privilege, so without one the check says it is skipped and CTest reports the test so.
void check_writer(const std::filesystem::path& dir) {
    const auto link = dir / "link", target = dir / "target";
#if defined(_WIN32)
#if !defined(SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)
#define SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE 0x2
#endif
    if (!CreateSymbolicLinkW(link.c_str(), target.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
        if (GetLastError() == ERROR_PRIVILEGE_NOT_HELD) {
            std::cout << "format-output: symbolic links need a privilege here, so the link check is skipped\n";
            return;
        }
        require(false, "cannot create a symbolic link");
    }
#else
    std::filesystem::create_symlink(target, link);
#endif
    bool caught = false;
    try {
        format::FileWriter writer(link.u8string(), false);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    require(caught, "a dangling symbolic link at the writer's path was followed");
    require(!std::filesystem::exists(target), "the writer created the link's target");
    std::filesystem::remove(link);
}

} // namespace

int main(int argc, char** argv) {
    const bool link = argc == 3 && std::string(argv[2]) == "--link";
    if (argc != 2 && !link) return 2;
    try {
        Directory dir(argv[1]);
        if (link) {
            check_writer(dir.path);
            std::cout << "format-output: the writer's link refusal passes\n";
            return 0;
        }
        check(dir.path);
        std::cout << "format-output: preparation, checked streams, publication and cleanup pass\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
