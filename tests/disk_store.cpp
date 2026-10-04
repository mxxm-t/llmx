// The disk tier's store alone (server::DiskStore, docs/DISK-TIER.md): entries written and read back bit for bit, checked, swept, adopted and refused.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "core/crc32c.hpp"
#include "server/disk_store.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

namespace fs = std::filesystem;
constexpr size_t kSlab = size_t(1) << 20;

// Whether entry `key`'s file is in place in the store's directory.
bool in_place(const server::DiskStore& store, uint64_t key) {
    return fs::exists(store.directory() / ("entry-" + std::to_string(key) + ".kv"));
}

// Runs of these byte counts over fresh host-visible slabs, filled from `seed` (or zero).
std::vector<server::StoreRun> runs_of(backend::CpuBackend& cpu, const std::vector<size_t>& bytes, uint32_t seed) {
    std::vector<server::StoreRun> runs;
    for (size_t b : bytes) {
        server::StoreRun r;
        r.bytes = b;
        for (size_t at = 0; at < b; at += kSlab) {
            r.slabs.push_back(cpu.alloc(kSlab, backend::Memory::host_visible));
            auto* p = (uint8_t*)const_cast<void*>(r.slabs.back()->host_ptr());
            for (size_t i = 0; i < kSlab; ++i) {
                seed = seed * 1664525u + 1013904223u;
                p[i] = seed ? (uint8_t)(seed >> 24) : 0;
            }
        }
        runs.push_back(std::move(r));
    }
    return runs;
}

std::vector<uint8_t> bytes_of(const std::vector<server::StoreRun>& runs) {
    std::vector<uint8_t> out;
    for (const auto& r : runs)
        for (size_t at = 0; at < r.bytes; at += kSlab) {
            const auto* p = (const uint8_t*)r.slabs[at / kSlab]->host_ptr();
            out.insert(out.end(), p, p + std::min(kSlab, r.bytes - at));
        }
    return out;
}

// A call's outcome.
struct Outcome {
    bool ok = false;
    std::string error;
};
// A store call's answer, waited for at most a minute, so a call that never ends fails the test instead of hanging it; the promise outlives the wait, as a late answer still sets it.
struct Call {
    std::shared_ptr<std::promise<Outcome>> p = std::make_shared<std::promise<Outcome>>();
    std::future<Outcome> f = p->get_future();
    server::DiskStore::Done done() const {
        auto q = p;
        return [q](bool ok, const std::string& e) { q->set_value({ok, e}); };
    }
    Outcome wait() {
        if (f.wait_for(std::chrono::seconds(60)) != std::future_status::ready) throw std::runtime_error("a store call did not end in 60 seconds");
        return f.get();
    }
};
Outcome wait_put(server::DiskStore& store, uint64_t& key, std::string blob, std::vector<server::StoreRun> runs) {
    Call c;
    key = store.put(std::move(blob), std::move(runs), kSlab, c.done());
    return c.wait();
}
Outcome wait_get(server::DiskStore& store, uint64_t key, std::vector<server::StoreRun> runs) {
    Call c;
    store.get(key, std::move(runs), kSlab, c.done());
    return c.wait();
}

std::array<uint8_t, 32> identity(uint8_t v) {
    std::array<uint8_t, 32> id{};
    id.fill(v);
    return id;
}

std::vector<fs::path> servers(const fs::path& root) {
    std::vector<fs::path> v;
    for (const auto& e : fs::directory_iterator(root))
        if (e.is_directory()) v.push_back(e.path());
    return v;
}

void age(const fs::path& file, int hours) {
    fs::last_write_time(file, fs::file_time_type::clock::now() - std::chrono::hours(hours));
}

} // namespace

int main(int argc, char** argv) {
    // As a child of the test: hold directory argv[2]'s lock until a file named stop appears in it, or two minutes pass, so a test that failed before writing stop leaves no process behind.
    // The directory's name is written whole and then renamed into place, so the test never reads it half written.
    if (argc == 3 && std::string(argv[1]) == "hold") {
        const fs::path dir = fs::u8path(argv[2]);
        server::DiskStore::Options o;
        o.root = argv[2];
        server::DiskStore held(o, identity(9));
        std::ofstream(dir / "held.tmp") << held.directory().u8string();
        fs::rename(dir / "held.tmp", dir / "held");
        const auto until = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        while (!fs::exists(dir / "stop") && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return 0;
    }
    try {
        if (argc != 2) throw std::runtime_error("usage: llmx-disk-store-test DIRECTORY");
        const fs::path base = fs::u8path(argv[1]);
        fs::remove_all(base);
        fs::create_directories(base);
        backend::CpuBackend cpu;
        require(core::crc32c(0, "123456789", 9) == 0xE3069283u, "CRC32C of the standard check string");

        // An entry of two runs over several slabs, not whole slabs, read back bit for bit with its blob, its file the size its layout gives; with keep it outlives the store, marked kept.
        const fs::path root = base / "root";
        const std::vector<size_t> layout{3 * kSlab + 12345, 5 * kSlab + 7};
        const auto written = runs_of(cpu, layout, 1);
        const std::vector<uint8_t> want = bytes_of(written);
        uint64_t key = 0;
        fs::path kept_dir;
        {
            server::DiskStore::Options o;
            o.root = root.u8string();
            o.keep = true;
            server::DiskStore store(o, identity(1));
            std::cout << "disk-store: writes " << (store.direct() ? "around" : "through") << " the file cache\n";
#if !defined(_WIN32)
            require(fs::status(store.directory()).permissions() == fs::perms::owner_all, "the store's directory is not its owner's alone");
#endif
            const Outcome put = wait_put(store, key, "the blob", written);
            require(put.ok, "an entry was not written: " + put.error);
            require(in_place(store, key), "a written entry is not there");
            require(fs::file_size(store.directory() / ("entry-" + std::to_string(key) + ".kv")) == server::DiskStore::file_bytes(8, layout),
                    "an entry's file is not the size its layout gives");
            auto back = runs_of(cpu, layout, 0);
            const Outcome got = wait_get(store, key, back);
            require(got.ok && bytes_of(back) == want, "an entry read back differs: " + got.error);
            // Another layout is refused and the entry deleted.
            uint64_t other_key = 0;
            require(wait_put(store, other_key, "x", runs_of(cpu, {kSlab}, 2)).ok, "a second entry was not written");
            require(!wait_get(store, other_key, runs_of(cpu, {kSlab + 1}, 0)).ok && !in_place(store, other_key), "an entry read into another layout was not refused and deleted");
            // A missing entry is refused.
            require(!wait_get(store, 999, runs_of(cpu, {kSlab}, 0)).ok, "a missing entry was read");
            // A write cancelled while queued behind another is not kept.
            uint64_t first = 0, second = 0;
            Call c1, c2;
            first = store.put("a", runs_of(cpu, {16 * kSlab}, 3), kSlab, c1.done());
            second = store.put("b", runs_of(cpu, {kSlab}, 4), kSlab, c2.done());
            require(store.cancel(second), "a queued write was not found to cancel");
            const Outcome o1 = c1.wait(), o2 = c2.wait();
            require(o1.ok && !o2.ok && o2.error == "cancelled" && !in_place(store, second) && !fs::exists(store.directory() / ("entry-" + std::to_string(second) + ".tmp")),
                    "a cancelled write left something: " + o2.error);
            store.evict(first);
            require(!in_place(store, first) && !fs::exists(store.directory() / ("entry-" + std::to_string(first) + ".kv")), "an evicted entry is still there");
            kept_dir = store.directory();
        }
        require(fs::exists(kept_dir / "kept") && fs::exists(kept_dir / ("entry-" + std::to_string(key) + ".kv")), "a store with keep did not leave its entries");

        // Another identity under keep adopts nothing and removes what it cannot read; the same identity adopts the entry, which reads back, and a store without keep removes its directory at exit.
        const fs::path copy = base / "copy";
        fs::copy(root, copy, fs::copy_options::recursive);
        {
            server::DiskStore::Options o;
            o.root = copy.u8string();
            o.keep = true;
            server::DiskStore other(o, identity(2));
            require(other.adopted().empty() && servers(copy).size() == 1, "a store of another identity adopted an entry or left the directory");
        }
        fs::remove_all(copy);
        {
            server::DiskStore::Options o;
            o.root = root.u8string();
            o.keep = true;
            server::DiskStore again(o, identity(1));
            require(again.adopted().size() == 1 && again.adopted()[0].blob == "the blob" && again.adopted()[0].run_bytes == layout && !fs::exists(kept_dir),
                    "the same identity did not adopt the entry");
            auto back = runs_of(cpu, layout, 0);
            require(wait_get(again, again.adopted()[0].key, back).ok && bytes_of(back) == want, "an adopted entry read back differs");
            // A payload byte flipped on disk fails its checksum and deletes the entry; a truncated file is refused.
            uint64_t k2 = 0, k3 = 0;
            require(wait_put(again, k2, "c", runs_of(cpu, layout, 5)).ok && wait_put(again, k3, "d", runs_of(cpu, layout, 6)).ok, "entries to damage were not written");
            const fs::path f2 = again.directory() / ("entry-" + std::to_string(k2) + ".kv"), f3 = again.directory() / ("entry-" + std::to_string(k3) + ".kv");
            {
                std::fstream f(f2, std::ios::in | std::ios::out | std::ios::binary);
                f.seekp((std::streamoff)(server::DiskStore::kAlign + 5 * kSlab));
                f.put('\x7f');
            }
            const Outcome flipped = wait_get(again, k2, runs_of(cpu, layout, 0));
            require(!flipped.ok && flipped.error.find("checksum") != std::string::npos && !in_place(again, k2) && !fs::exists(f2), "a flipped byte was not caught: " + flipped.error);
            fs::resize_file(f3, server::DiskStore::kAlign + kSlab);
            require(!wait_get(again, k3, runs_of(cpu, layout, 0)).ok && !in_place(again, k3), "a truncated entry was read");
            kept_dir = again.directory();
        }
        {
            server::DiskStore::Options o;
            o.root = root.u8string();
            server::DiskStore plain(o, identity(1));
            require(plain.adopted().empty(), "a store without keep adopted an entry");
            kept_dir = plain.directory();
        }
        require(!fs::exists(kept_dir), "a store without keep left its directory");

        // A crashed server's directory, unlocked with a temporary file and an entry in it, is removed by the next start; a kept one is kept until it passes the age limit, and adoption leaves out entries past it.
        fs::remove_all(root);
        fs::create_directories(root / "server-1-1");
        std::ofstream(root / "server-1-1" / "entry-1.tmp") << "torn";
        std::ofstream(root / "server-1-1" / "entry-2.kv") << "junk";
        fs::create_directories(root / "server-2-2");
        std::ofstream(root / "server-2-2" / "kept") << "\n";
        fs::create_directories(root / "server-3-3");
        std::ofstream(root / "server-3-3" / "kept") << "\n";
        age(root / "server-3-3" / "kept", 48);
        {
            server::DiskStore::Options o;
            o.root = root.u8string();
            o.max_age = 24 * 3600;
            server::DiskStore store(o, identity(1));
            require(!fs::exists(root / "server-1-1") && fs::exists(root / "server-2-2") && !fs::exists(root / "server-3-3"),
                    "the sweep did not remove a crashed directory and an expired kept one, keeping a young kept one");
        }
        fs::remove_all(root);
        {
            server::DiskStore::Options o;
            o.root = root.u8string();
            o.keep = true;
            uint64_t young = 0, old = 0;
            {
                server::DiskStore s(o, identity(1));
                require(wait_put(s, young, "young", runs_of(cpu, {kSlab}, 7)).ok && wait_put(s, old, "old", runs_of(cpu, {kSlab}, 8)).ok, "entries to age were not written");
                age(s.directory() / ("entry-" + std::to_string(old) + ".kv"), 48);
            }
            o.max_age = 24 * 3600;
            server::DiskStore s(o, identity(1));
            require(s.adopted().size() == 1 && s.adopted()[0].blob == "young", "adoption took an entry past the age limit");
        }

        // A live server's directory is never touched: a child holds its lock while a store starts beside it.
        fs::remove_all(root);
        fs::create_directories(root);
        {
            const std::string exe = fs::absolute(fs::u8path(argv[0])).u8string();
            // The child writes nothing and keeps none of the test's output open, which CTest would otherwise wait on, and is told to stop however this case ends.
#if defined(_WIN32)
            const std::string command = "start \"\" /b \"" + exe + "\" hold \"" + root.u8string() + "\" >NUL 2>&1";
#else
            const std::string command = "\"" + exe + "\" hold \"" + root.u8string() + "\" >/dev/null 2>&1 </dev/null &";
#endif
            struct Stop {
                fs::path file;
                ~Stop() { std::ofstream(file) << "\n"; }
            } stop{root / "stop"};
            require(std::system(command.c_str()) == 0, "the child holding a lock did not start");
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!fs::exists(root / "held")) {
                require(std::chrono::steady_clock::now() < until, "the child did not lock its directory in 30 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::string held;
            std::getline(std::ifstream(root / "held"), held);
            {
                server::DiskStore::Options o;
                o.root = root.u8string();
                server::DiskStore beside(o, identity(1));
                require(fs::exists(fs::u8path(held)), "a store's sweep removed a live server's directory");
            }
            std::ofstream(root / "stop") << "\n";
            const auto gone = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (fs::exists(fs::u8path(held))) {
                require(std::chrono::steady_clock::now() < gone, "the child did not end in 30 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }

        // The free-space floor stops a write, which leaves nothing.
        {
            server::DiskStore::Options o;
            o.root = root.u8string();
            o.floor = uint64_t(1) << 62;
            server::DiskStore store(o, identity(1));
            uint64_t k = 0;
            const Outcome floor = wait_put(store, k, "e", runs_of(cpu, {kSlab}, 9));
            require(!floor.ok && floor.error.find("floor") != std::string::npos && !in_place(store, k), "the floor did not stop a write: " + floor.error);
        }
        fs::remove_all(base);
        std::cout << "disk-store: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "disk-store: " << e.what() << '\n';
        return 1;
    }
}
