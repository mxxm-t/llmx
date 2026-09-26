// format::FileReader and core::HostPages: reads at any offset and length give the file's bytes, a read past the end is short by exactly what the file lacks, empty and tiny files read, several threads reading one reader at once each get their own range, direct reads where the file system takes them, and reserved pages committed and decommitted by range.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "core/host_memory.hpp"
#include "format/file_reader.hpp"

namespace {

int checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

std::vector<uint8_t> pattern(size_t bytes) {
    std::vector<uint8_t> v(bytes);
    uint32_t x = 0x12345678u;
    for (auto& b : v) {
        x = x * 1664525u + 1013904223u;
        b = (uint8_t)(x >> 24);
    }
    return v;
}

void save(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write((const char*)bytes.data(), (std::streamsize)bytes.size());
    require((bool)out, "cannot write fixture " + path.u8string());
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: llmx-file-reader-test DIRECTORY");
        const std::filesystem::path dir = std::filesystem::u8path(argv[1]);
        std::filesystem::create_directories(dir);

        const core::HostPages pages(1);
        require(pages.size() == core::page_size() && (uintptr_t)pages.data() % core::page_size() == 0, "host pages are not whole aligned pages");
        core::HostPages moved(3 * core::page_size() + 1);
        require(moved.size() == 4 * core::page_size(), "host pages were not rounded up to whole pages");
        uint8_t* held = moved.data();
        core::HostPages taken(std::move(moved));
        require(taken.data() == held && !moved.data() && !moved.size(), "moving host pages did not hand them over");

        // Reserved pages: address space rounded to pages, memory for a committed range only, which may start and end inside a page and be committed again, and back to the reservation on decommit.
        {
            const size_t page = core::page_size();
            const core::HostPages none = core::HostPages::reserved(0);
            require(!none.data() && !none.size(), "an empty reservation holds address space");
            core::HostPages r = core::HostPages::reserved(5 * page + 1);
            require(r.data() && r.size() == 6 * page && (uintptr_t)r.data() % page == 0, "a reservation is not whole aligned pages");
            r.commit(page - 1, 2);
            r.data()[page - 1] = 7;
            r.data()[page] = 9;
            r.commit(page - 1, page + 2);
            r.data()[2 * page] = 11;
            require(r.data()[page - 1] == 7 && r.data()[page] == 9 && r.data()[2 * page] == 11, "committing a range again lost what it held");
            for (const auto& [off, len] : std::vector<std::pair<size_t, size_t>>{{r.size(), 1}, {0, r.size() + 1}}) {
                bool outside = false;
                try { r.commit(off, len); } catch (const std::logic_error&) { outside = true; }
                require(outside, "a commit outside the reservation was taken");
            }
            r.decommit(page / 2, 2 * page);   // gives back the one whole page inside, keeping the two it covers in part
            require(r.data()[page - 1] == 7 && r.data()[2 * page] == 11, "a decommit took a page it covers only in part");
            r.commit(page, page);
            r.data()[page] = 3;
            require(r.data()[page] == 3, "a decommitted page could not be committed again");
            core::HostPages moved_r(std::move(r));
            require(moved_r.size() == 6 * page && !r.data(), "moving a reservation did not hand it over");
        }

        // A file of several granules, the file system's larger than 1 MiB included, and an odd tail.
        size_t largest = 1024 * 1024;
        {
            const auto probe = dir / "file-reader-probe.bin";
            save(probe, pattern(4096));
            largest = std::max(largest, format::FileReader(probe.u8string()).granule());
            try { largest = std::max(largest, format::FileReader(probe.u8string(), true).granule()); } catch (const format::DirectUnavailable&) {}
        }
        const auto big = pattern(3 * largest + 4097);
        const auto path = dir / "file-reader-big.bin";
        save(path, big);
        {
            const format::FileReader r(path.u8string());
            require(r.size() == big.size(), "the reader gives the wrong size");
            require(r.granule() >= core::page_size() && r.granule() % core::page_size() == 0, "the granule is not a multiple of the page");
            core::HostPages buf(big.size() + r.granule());
            for (const auto& [off, len] : std::vector<std::pair<uint64_t, size_t>>{
                     {0, big.size()}, {0, r.granule()}, {r.granule(), 2 * r.granule()}, {1, 1}, {4095, 4098}, {big.size() - 1, 1}}) {
                const size_t n = r.read(off, buf.data(), len);
                require(n == len && std::memcmp(buf.data(), big.data() + off, len) == 0,
                        "a read of " + std::to_string(len) + " bytes at " + std::to_string(off) + " differs from the file");
            }
            // Past the end: short by what the file lacks, and nothing at or beyond it.
            require(r.read(big.size() - 10, buf.data(), 4096) == 10 && std::memcmp(buf.data(), big.data() + big.size() - 10, 10) == 0,
                    "a read past the end is not short by what the file lacks");
            require(r.read(big.size(), buf.data(), 4096) == 0 && r.read(big.size() + 4096, buf.data(), 1) == 0,
                    "a read at or past the end returned bytes");
            // Eight threads at once, each its own stripe, several times over.
            const size_t stripe = big.size() / 8;
            std::vector<std::thread> threads;
            std::vector<int> ok(8, 0);
            for (size_t t = 0; t < 8; ++t) {
                threads.emplace_back([&, t] {
                    core::HostPages mine(stripe + 4096);
                    int good = 0;
                    for (int round = 0; round < 16; ++round) {
                        const uint64_t off = t * stripe + (uint64_t)round * 7;
                        const size_t len = stripe - (size_t)round * 7;
                        good += r.read(off, mine.data(), len) == len && std::memcmp(mine.data(), big.data() + off, len) == 0;
                    }
                    ok[t] = good;
                });
            }
            for (auto& th : threads) th.join();
            for (int good : ok) require(good == 16, "a concurrent read differs from the file");
        }
        // Tiny and empty files.
        for (size_t bytes : {size_t(0), size_t(1), size_t(7)}) {
            const auto small = pattern(bytes);
            const auto p = dir / ("file-reader-" + std::to_string(bytes) + ".bin");
            save(p, small);
            const format::FileReader r(p.u8string());
            core::HostPages buf(4096);
            require(r.size() == bytes && r.read(0, buf.data(), 4096) == bytes && (!bytes || std::memcmp(buf.data(), small.data(), bytes) == 0),
                    "a " + std::to_string(bytes) + "-byte file did not read whole");
        }
        bool refused = false;
        try { format::FileReader r((dir / "file-reader-missing.bin").u8string()); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "a missing file was opened");
        // Direct reads where this file system takes them: aligned ranges read the file, a read rounded up past the end is short by what the file lacks, and a misaligned request is refused rather than served.
        // Where it does not, opening for direct reads says so.
        try {
            const format::FileReader r(path.u8string(), true);
            const size_t g = r.granule();
            require(r.direct() && g >= core::page_size() && g % core::page_size() == 0, "the direct granule is not a multiple of the page");
            core::HostPages buf(big.size() + 2 * g);
            require(r.read(0, buf.data(), 2 * g) == 2 * g && std::memcmp(buf.data(), big.data(), 2 * g) == 0, "a direct read differs from the file");
            const uint64_t last = big.size() / g * g;
            const size_t tail = big.size() - size_t(last);
            require(r.read(last, buf.data(), g) == tail && std::memcmp(buf.data(), big.data() + last, tail) == 0,
                    "a direct read rounded past the end is not short by what the file lacks");
            for (const auto& [off, len, at] : std::vector<std::tuple<uint64_t, size_t, size_t>>{{1, g, 0}, {0, g + 1, 0}, {0, g, 1}}) {
                bool misaligned = false;
                try { r.read(off, buf.data() + at, len); } catch (const std::logic_error&) { misaligned = true; }
                require(misaligned, "a misaligned direct read was served");
            }
            std::cout << "file-reader: direct reads, granule " << g << "\n";
        } catch (const format::DirectUnavailable& e) {
            require(std::string(e.what()).find(path.u8string()) != std::string::npos, "a direct refusal does not name its file");
            std::cout << "file-reader: direct reads refused here: " << e.what() << "\n";
        }
        std::cout << "file-reader: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "file-reader: " << e.what() << '\n';
        return 1;
    }
}
