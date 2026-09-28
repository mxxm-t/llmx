#pragma once
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace format {

// A conversion output prepared beside its destination and renamed only after checked close.
// Destruction removes only this object's temporary file/directory, never the destination.
class OutputFile {
public:
    explicit OutputFile(const std::string& path, std::ios::openmode mode = std::ios::binary)
        : name_(path), destination_(std::filesystem::u8path(path)) {
        try {
            check_destination();
            auto parent = destination_.parent_path();
            if (parent.empty()) parent = ".";
            std::random_device random;
            for (unsigned attempt = 0; attempt < 32; ++attempt) {
                auto candidate = parent / (".llmx-output-" + std::to_string(random()) + "-" + std::to_string(random()));
                if (std::filesystem::create_directory(candidate)) {
                    directory_ = std::move(candidate);
                    break;
                }
            }
            if (directory_.empty()) throw std::runtime_error("cannot reserve temporary directory");
            temporary_ = directory_ / "output";
            stream_.exceptions(std::ios::failbit | std::ios::badbit);
            stream_.open(temporary_, mode);
        } catch (const std::system_error& e) {
            cleanup();
            throw std::runtime_error("cannot prepare output " + name_ + ": " + system_reason(e));
        } catch (const std::exception& e) {
            cleanup();
            throw std::runtime_error("cannot prepare output " + name_ + ": " + e.what());
        }
    }
    ~OutputFile() { cleanup(); }
    OutputFile(const OutputFile&) = delete;
    OutputFile& operator=(const OutputFile&) = delete;

    // Serialization and buffered completion must both succeed before publish can expose the file.
    template<class Write> void write(Write&& write) {
        try {
            write(stream_);
            stream_.close();
            complete_ = true;
        } catch (const std::ios_base::failure& e) {
            throw std::runtime_error("cannot write output " + name_ + ": " + system_reason(e));
        }
    }

    void publish() {
        if (!complete_) throw std::logic_error("output is not complete: " + name_);
        try {
            check_destination();
            std::filesystem::rename(temporary_, destination_);
        } catch (const std::system_error& e) {
            throw std::runtime_error("cannot publish output " + name_ + ": " + system_reason(e));
        } catch (const std::exception& e) {
            throw std::runtime_error("cannot publish output " + name_ + ": " + e.what());
        }
    }

private:
    // A filesystem exception's what() can embed Windows code-page paths. Keep our UTF-8 name and a stable category/code.
    static std::string system_reason(const std::system_error& error) {
        return std::string(error.code().category().name()) + " error " + std::to_string(error.code().value());
    }

    void check_destination() const {
        std::error_code ec;
        const auto status = std::filesystem::symlink_status(destination_, ec);
        if (ec && ec != std::errc::no_such_file_or_directory)
            throw std::system_error(ec);
        if (std::filesystem::exists(status) && !std::filesystem::is_regular_file(status))
            throw std::runtime_error("destination is not a regular file");
    }

    void cleanup() noexcept {
        // A failed close can leave the stream open. Turn off exceptions before the final cleanup attempt.
        stream_.exceptions(std::ios::goodbit);
        if (stream_.is_open()) stream_.close();
        std::error_code ec;
        if (!temporary_.empty()) std::filesystem::remove(temporary_, ec);
        if (!directory_.empty()) std::filesystem::remove(directory_, ec);
    }

    std::string name_;
    std::filesystem::path destination_, directory_, temporary_;
    std::ofstream stream_;
    bool complete_ = false;
};

} // namespace format
