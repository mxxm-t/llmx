#pragma once
// A file mapped read-only into memory, unmapped when the last holder lets it go.
// Its pages are the operating system's to evict and read back, so a model larger than what the host can hold beside it still loads: pages a device has copied are simply not touched again.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace format {

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
#if defined(_WIN32)
        const std::wstring wide = std::filesystem::u8path(path).wstring();
        file_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open file: " + path);
        LARGE_INTEGER size;
        if (!GetFileSizeEx(file_, &size)) { close(); throw std::runtime_error("cannot size file: " + path); }
        size_ = (size_t)size.QuadPart;
        if (size_) {
            map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (map_) data_ = (const uint8_t*)MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
            if (!data_) { close(); throw std::runtime_error("cannot map file: " + path); }
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open file: " + path);
        struct stat st {};
        if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("cannot size file: " + path); }
        size_ = (size_t)st.st_size;
        if (size_) {
            void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p == MAP_FAILED) { ::close(fd); throw std::runtime_error("cannot map file: " + path); }
            data_ = (const uint8_t*)p;
        }
        ::close(fd);
#endif
    }
    ~MappedFile() { close(); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

    // Tell the OS that `bytes` from `p` will not be read again soon, so their pages go before any other the host still reads; a later read brings them back from the file.
    // Only whole pages inside the range, so a neighbour sharing a boundary page keeps it.
    void drop(const void* p, size_t bytes) const {
        const size_t page = page_size();
        const uintptr_t lo = ((uintptr_t)p + page - 1) / page * page, hi = ((uintptr_t)p + bytes) / page * page;
        if (hi <= lo || lo < (uintptr_t)data_ || hi > (uintptr_t)data_ + size_) return;
#if defined(_WIN32)
        // Unlocking pages that were never locked takes them out of the working set.
        VirtualUnlock((void*)lo, hi - lo);
#else
        madvise((void*)lo, hi - lo, MADV_DONTNEED);
#endif
    }

private:
    static size_t page_size() {
#if defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t)si.dwPageSize;
#else
        return (size_t)sysconf(_SC_PAGESIZE);
#endif
    }

    void close() {
#if defined(_WIN32)
        if (data_) UnmapViewOfFile(data_);
        if (map_) CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        map_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (data_) munmap((void*)data_, size_);
#endif
        data_ = nullptr;
    }

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
#endif
};

} // namespace format
