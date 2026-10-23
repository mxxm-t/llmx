#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "format/file_reader.hpp"
#include "model/weight_rows.hpp"

// Allocations fail while this is set, so a start can be made to fail once its checks have passed.
std::atomic<bool> fail_allocations{false};
void* operator new(std::size_t n) {
    if (fail_allocations.load()) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

int checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

std::vector<uint8_t> payload(size_t rows, size_t bytes_per_row) {
    std::vector<uint8_t> out(rows * bytes_per_row);
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < bytes_per_row; ++i)
            out[r * bytes_per_row + i] = uint8_t((r * 37 + i * 13) & 255);
    return out;
}

void save(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write((const char*)bytes.data(), (std::streamsize)bytes.size());
    require(bool(out), "cannot write " + path.string());
}

// A read started and finished at once, as a stage does across its layers (Model::start_lookup, Model::finish_lookup).
infer::WeightRows::Rows read_rows(const infer::WeightRows& source, const uint32_t* ids, size_t count, infer::WeightRows::Scratch& scratch) {
    source.start(ids, count, scratch);
    return source.finish(scratch);
}

void check_rows(const infer::WeightRows& source, const std::vector<uint8_t>& original,
                size_t row_bytes, const std::vector<uint32_t>& ids, size_t capacity) {
    infer::WeightRows::Scratch scratch(source, capacity);
    const infer::WeightRows::Rows got = read_rows(source, ids.data(), ids.size(), scratch);
    std::vector<uint32_t> unique = ids;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    require(got.unique == unique.size() && got.requested == ids.size() && got.row_bytes == row_bytes,
            "read reports the wrong row counts or width");
    for (size_t i = 0; i < unique.size(); ++i) {
        require(got.ids[i] == unique[i], "unique row IDs are not sorted");
        require(std::equal(got.data + i * row_bytes, got.data + (i + 1) * row_bytes,
                           original.data() + size_t(unique[i]) * row_bytes),
                "packed unique rows are not sorted raw source rows");
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        const size_t mapped = got.remap[i];
        require(mapped < got.unique && unique[mapped] == ids[i], "request row remap lost order or duplicates");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: llmx-weight-rows-test DIRECTORY");
        const std::filesystem::path dir = std::filesystem::u8path(argv[1]);
        std::filesystem::create_directories(dir);

        const size_t rows = 9, row_bytes = quant::row_bytes(quant::GGML_TYPE_Q8_0, 160);
        const auto data = payload(rows, row_bytes);
        const std::vector<uint32_t> ids{8, 1, 8, 2, 4, 1};
        const infer::WeightRows memory(quant::GGML_TYPE_Q8_0, 160, rows, data.data(), data.size());
        check_rows(memory, data, row_bytes, ids, 8);

        std::weak_ptr<std::vector<uint8_t>> memory_owner_lifetime;
        std::unique_ptr<infer::WeightRows::Scratch> memory_owner_scratch;
        const std::vector<uint32_t> owned_ids{5, 0, 5};
        {
            auto allocation = std::make_shared<std::vector<uint8_t>>(data);
            memory_owner_lifetime = allocation;
            std::shared_ptr<const void> owner = allocation;
            const infer::WeightRows owned_memory(quant::GGML_TYPE_Q8_0, 160, rows,
                                                 allocation->data(), allocation->size(), owner);
            memory_owner_scratch = std::make_unique<infer::WeightRows::Scratch>(owned_memory, owned_ids.size());
            allocation.reset();
            owner.reset();
            require(!memory_owner_lifetime.expired(), "memory source did not retain its shared owner");
            const auto got = read_rows(owned_memory, owned_ids.data(), owned_ids.size(), *memory_owner_scratch);
            require(got.unique == 2 && got.remap[0] == 1 && got.remap[1] == 0 && got.remap[2] == 1,
                    "owned memory source lost request order or duplicates");
            require(got.ids[0] == 0 && got.ids[1] == 5 &&
                    std::equal(got.data, got.data + row_bytes, data.data()) &&
                    std::equal(got.data + row_bytes, got.data + 2 * row_bytes, data.data() + 5 * row_bytes),
                    "owned memory source did not read after the caller dropped its owner");
        }
        require(memory_owner_lifetime.expired(), "scratch unexpectedly retained the memory source owner");
        memory_owner_scratch.reset();

        bool bad_width = false;
        try { infer::WeightRows bad(quant::GGML_TYPE_Q8_0, 159, rows, data.data(), data.size()); }
        catch (const std::runtime_error&) { bad_width = true; }
        require(bad_width, "a partial quantized row was accepted");
        bool bad_extent = false;
        try { infer::WeightRows bad(quant::GGML_TYPE_Q8_0, 160, rows, data.data(), data.size() - 1); }
        catch (const std::invalid_argument&) { bad_extent = true; }
        require(bad_extent, "a memory extent that disagrees with its shape was accepted");
        bool zero_columns = false;
        try { infer::WeightRows bad(quant::GGML_TYPE_Q8_0, 0, 1, nullptr, 0); }
        catch (const std::invalid_argument&) { zero_columns = true; }
        require(zero_columns, "a table with zero columns was accepted");
        bool ids_overflow = false;
        try { infer::WeightRows bad(quant::GGML_TYPE_Q8_0, 160, size_t(uint64_t(UINT32_MAX) + 2), data.data(), SIZE_MAX); }
        catch (const std::invalid_argument&) { ids_overflow = true; }
        require(ids_overflow, "a table with rows outside the ID range was accepted");

        infer::WeightRows::Scratch scratch(memory, ids.size());
        const auto empty = read_rows(memory, nullptr, 0, scratch);
        require(!empty.data && !empty.ids && !empty.remap && !empty.unique && !empty.requested && empty.row_bytes == row_bytes,
                "an empty null-ID request has the wrong result");
        const auto previous = read_rows(memory, ids.data(), ids.size(), scratch);
        std::vector<uint8_t> before(previous.data, previous.data + previous.unique * row_bytes);
        bool invalid = false;
        try {
            const uint32_t bad_ids[] = {3, uint32_t(rows)};
            read_rows(memory, bad_ids, 2, scratch);
        } catch (const std::out_of_range&) { invalid = true; }
        require(invalid, "an out-of-range row id was accepted");
        require(std::equal(before.begin(), before.end(), previous.data), "an invalid id changed row output before refusal");
        bool over_capacity = false;
        infer::WeightRows::Scratch short_capacity(memory, 2);
        const uint32_t small_ids[] = {2, 3};
        const auto small = read_rows(memory, small_ids, 2, short_capacity);
        std::vector<uint8_t> small_before(small.data, small.data + 2 * row_bytes);
        try { read_rows(memory, ids.data(), ids.size(), short_capacity); }
        catch (const std::length_error&) { over_capacity = true; }
        require(over_capacity, "a request past scratch capacity was accepted");
        require(std::equal(small_before.begin(), small_before.end(), small.data), "an over-capacity request changed row output");
        bool missing_ids = false;
        try { read_rows(memory, nullptr, 1, scratch); }
        catch (const std::invalid_argument&) { missing_ids = true; }
        require(missing_ids, "a nonempty null-ID request was accepted");
        const auto file_bytes = [&] {
            std::vector<uint8_t> v(37, 0xa5);
            v.insert(v.end(), data.begin(), data.end());
            v.insert(v.end(), 29, 0x5a);
            return v;
        }();
        const auto path = dir / "weight-rows.bin";
        save(path, file_bytes);
        auto reader = std::make_shared<const format::FileReader>(path.u8string());
        const infer::WeightRows file(quant::GGML_TYPE_Q8_0, 160, rows, reader, 37, data.size());
        check_rows(file, data, row_bytes, ids, 8);
        require(file.staging_bytes() == 0 && memory.staging_bytes() == 0, "a buffered or memory source counts I/O windows it does not hold");
        bool bad_file_extent = false;
        try { infer::WeightRows bad(quant::GGML_TYPE_Q8_0, 160, rows, reader, 37, data.size() + 1); }
        catch (const std::invalid_argument&) { bad_file_extent = true; }
        require(bad_file_extent, "a file span beyond the reader extent was accepted");
        bool incompatible = false;
        try { read_rows(file, ids.data(), ids.size(), scratch); }
        catch (const std::invalid_argument&) { incompatible = true; }
        require(incompatible, "memory scratch was reused for a file source");

        // A row wider than the I/O window is assembled from bounded reads.
        const size_t wide_row = quant::row_bytes(quant::GGML_TYPE_Q8_0, 4096);
        const auto wide = payload(4, wide_row);
        const infer::WeightRows wide_memory(quant::GGML_TYPE_Q8_0, 4096, 4, wide.data(), wide.size());
        check_rows(wide_memory, wide, wide_row, {3, 0, 3}, 3);
        const auto wide_path = dir / "weight-rows-wide.bin";
        std::vector<uint8_t> wide_file(11, 0x7c);
        wide_file.insert(wide_file.end(), wide.begin(), wide.end());
        save(wide_path, wide_file);
        auto wide_reader = std::make_shared<const format::FileReader>(wide_path.u8string());
        const infer::WeightRows wide_source(quant::GGML_TYPE_Q8_0, 4096, 4, wide_reader, 11, wide.size());
        check_rows(wide_source, wide, wide_row, {3, 0, 3}, 3);

        // A read started ahead gives at finish the file's rows, from a file and from memory; finish without a start is refused.
        {
            infer::WeightRows::Scratch ahead(file, ids.size());
            file.start(ids.data(), ids.size(), ahead);
            const auto got = file.finish(ahead);
            bool same = got.unique == 4 && got.requested == ids.size();
            for (size_t i = 0; i < ids.size() && same; ++i)
                same = std::equal(got.data + got.remap[i] * row_bytes, got.data + (got.remap[i] + 1) * row_bytes, data.data() + size_t(ids[i]) * row_bytes);
            require(same, "a file read started ahead gave other rows at finish");
            bool unstarted = false;
            try { file.finish(ahead); }
            catch (const std::logic_error&) { unstarted = true; }
            require(unstarted, "finish without a started read was accepted");
            // A read started and never finished is dropped by the next start, whose rows are its own.
            const uint32_t first[] = {0, 1}, second[] = {6, 7, 6};
            file.start(first, 2, ahead);
            file.start(second, 3, ahead);
            const auto next = file.finish(ahead);
            require(next.unique == 2 && next.ids[0] == 6 && next.ids[1] == 7 && next.remap[0] == 0 && next.remap[2] == 0 &&
                        std::equal(next.data, next.data + 2 * row_bytes, data.data() + 6 * row_bytes),
                    "a read started over an abandoned one did not give its own rows");
            infer::WeightRows::Scratch in_memory(memory, ids.size());
            memory.start(ids.data(), ids.size(), in_memory);
            const auto mem = memory.finish(in_memory);
            require(mem.unique == 4 && std::equal(mem.data, mem.data + row_bytes, data.data() + 1 * row_bytes), "a memory read started ahead gave other rows");
            // A start that fails once its checks have passed, here at its first allocation, leaves finish refusing, never giving the last read's rows under the new ids.
            infer::WeightRows::Scratch failing(file, ids.size());
            read_rows(file, ids.data(), ids.size(), failing);
            const uint32_t others[] = {0, 5};
            bool start_failed = false;
            fail_allocations = true;
            try { file.start(others, 2, failing); } catch (const std::bad_alloc&) { start_failed = true; }
            fail_allocations = false;
            require(start_failed, "the injected allocation failure did not fail the start");
            bool refused = false;
            try { file.finish(failing); } catch (const std::logic_error&) { refused = true; }
            require(refused, "finish after a failed start gave the last read's rows");
            // A scratch dropped with its read in flight waits for it.
            infer::WeightRows::Scratch dropped(file, ids.size());
            file.start(ids.data(), ids.size(), dropped);
        }

        // Many spans, more than the read threads, each read into its own place: a table of 600 rows asked for every third row and some twice.
        {
            const size_t many = 600;
            const auto table = payload(many, row_bytes);
            std::vector<uint8_t> table_file(5, 0x33);
            table_file.insert(table_file.end(), table.begin(), table.end());
            const auto table_path = dir / "weight-rows-many.bin";
            save(table_path, table_file);
            auto table_reader = std::make_shared<const format::FileReader>(table_path.u8string());
            const infer::WeightRows table_source(quant::GGML_TYPE_Q8_0, 160, many, table_reader, 5, table.size());
            std::vector<uint32_t> spread;
            for (uint32_t r = 0; r < many; r += 3) spread.push_back(r);
            for (uint32_t r = 0; r < many; r += 30) spread.push_back(r + 1);
            for (uint32_t r = 0; r < 40; ++r) spread.push_back(r * 7 % many);
            std::reverse(spread.begin(), spread.end());
            check_rows(table_source, table, row_bytes, spread, spread.size());
        }

        std::vector<std::thread> threads;
        std::vector<int> thread_ok(4, 0);
        for (size_t t = 0; t < thread_ok.size(); ++t) {
            threads.emplace_back([&, t] {
                try {
                    infer::WeightRows::Scratch local(file, ids.size());
                    const auto got = read_rows(file, ids.data(), ids.size(), local);
                    bool same = got.unique == 4 && got.remap[0] == got.remap[2] && got.remap[1] != got.remap[0];
                    for (size_t i = 0; i < got.unique && same; ++i)
                        same = got.ids[i] < rows && std::equal(got.data + i * row_bytes, got.data + (i + 1) * row_bytes,
                                                               data.data() + size_t(got.ids[i]) * row_bytes);
                    for (size_t i = 0; i < ids.size() && same; ++i)
                        same = got.remap[i] < got.unique && got.ids[got.remap[i]] == ids[i];
                    thread_ok[t] = same;
                } catch (...) {}
            });
        }
        for (auto& thread : threads) thread.join();
        for (int ok : thread_ok) require(ok == 1, "concurrent readers with separate scratch disagreed");

        // A shrink after construction must not publish partial rows.
#if !defined(_WIN32)
        const auto short_path = dir / "weight-rows-short.bin";
        save(short_path, file_bytes);
        auto short_reader = std::make_shared<const format::FileReader>(short_path.u8string());
        const infer::WeightRows short_source(quant::GGML_TYPE_Q8_0, 160, rows, short_reader, 37, data.size());
        std::filesystem::resize_file(short_path, 37 + data.size() - 1);
        infer::WeightRows::Scratch short_scratch(short_source, ids.size());
        bool short_read = false;
        try { read_rows(short_source, ids.data(), ids.size(), short_scratch); }
        catch (const std::runtime_error&) { short_read = true; }
        require(short_read, "a short file read was accepted");
        // Started ahead, the short read fails at finish and leaves no rows.
        short_source.start(ids.data(), ids.size(), short_scratch);
        bool short_finish = false;
        try { short_source.finish(short_scratch); }
        catch (const std::runtime_error&) { short_finish = true; }
        require(short_finish, "a short file read started ahead was accepted at finish");
#endif

        // Direct reads are checked only on file systems that support them.
        try {
            const auto eof_path = dir / "weight-rows-direct-eof.bin";
            std::vector<uint8_t> eof_file(37, 0xa5);
            eof_file.insert(eof_file.end(), data.begin(), data.end());
            save(eof_path, eof_file);
            auto direct = std::make_shared<const format::FileReader>(eof_path.u8string(), true);
            const infer::WeightRows direct_source(quant::GGML_TYPE_Q8_0, 160, rows, direct, 37, data.size());
            check_rows(direct_source, data, row_bytes, ids, 8);
            require(direct_source.staging_bytes() == infer::WeightRows::kReadThreads * direct->granule(),
                    "a direct source does not count an aligned window for each of its readers");

            if (direct->granule() <= (size_t(1) << 20)) {
                const size_t g = direct->granule();
                const size_t columns = (g / quant::Q8_0_TYPESIZE + 1) * quant::Q8_0_BLOCK;
                const size_t large_row = quant::row_bytes(quant::GGML_TYPE_Q8_0, columns);
                const auto large = payload(2, large_row);
                std::vector<uint8_t> large_file(11, 0x7c);
                large_file.insert(large_file.end(), large.begin(), large.end());
                large_file.resize(large_file.size() + 2 * g, 0x55);
                const auto large_path = dir / "weight-rows-direct-wide.bin";
                save(large_path, large_file);
                auto large_reader = std::make_shared<const format::FileReader>(large_path.u8string(), true);
                const infer::WeightRows large_source(quant::GGML_TYPE_Q8_0, columns, 2, large_reader, 11, large.size());
                check_rows(large_source, large, large_row, {1, 0}, 2);
            }
            std::cout << "weight-rows: direct reads, granule " << direct->granule() << "\n";
        } catch (const format::DirectUnavailable& e) {
            require(std::string(e.what()).find((dir / "weight-rows-direct-eof.bin").u8string()) != std::string::npos,
                    "direct-read refusal does not name the file");
            std::cout << "weight-rows: direct reads unavailable here\n";
        }

        std::cout << "weight-rows: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "weight-rows: FAIL: " << e.what() << "\n";
        return 1;
    }
}
