#pragma once
// A tensor group's collective over Vulkan devices and VulkanBackend's part in it; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#endif
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_device.hpp"
#include "core/job_threads.hpp"

namespace backend {
namespace {

// Partial, scratch and per-peer inbox rows for both parities; two-shot sums also reserve each peer's gather share.
// Imports consume no memory on the importing device.
inline size_t VulkanBackend::collective_bytes_per_row(size_t members, size_t width) const {
    if (!dev_->exchange || !members) return 0;
    const size_t rows = size_add(size_add(2, size_mul(2, members - 1)), two_shot_crossover(members) != SIZE_MAX ? 2 : 0);
    return size_mul(rows, size_mul(width, sizeof(float)));
}

// The measured crossover for two-shot sums on this transport (docs/STATUS.md, a group's large sums in two shots).
// Two members keep broadcast because two shots move the same bytes with another wait round.
inline size_t VulkanBackend::two_shot_crossover(size_t members) {
    return members < 3 ? SIZE_MAX : 327680;
}

// Preserve a copied peer range across the collective's semaphore exchange (docs/TENSOR-SPLIT.md, section 4.3).
inline VkBufferMemoryBarrier VulkanBackend::external_ownership_barrier(Buffer& buffer, size_t offset, size_t bytes, bool acquire) {
    VulkanBuffer& b = as_vulkan(buffer);
    span(b, offset, bytes);
    VkBufferMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    mb.srcAccessMask = acquire ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = acquire ? VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT : 0;
    mb.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_FOREIGN_EXT : dev_->queue_family;
    mb.dstQueueFamilyIndex = acquire ? dev_->queue_family : VK_QUEUE_FAMILY_FOREIGN_EXT;
    mb.buffer = b.handle();
    mb.offset = offset;
    mb.size = bytes;
    return mb;
}

// A peer's copy into memory this device imported is recorded on the peer, so the acquire is where this device learns the range changed.
inline void VulkanBackend::external_ownership(const VkBufferMemoryBarrier* barriers, size_t count, bool acquire) {
    if (!count) return;
    for (size_t i = 0; acquire && i < count; ++i) forget_written(barriers[i].buffer, barriers[i].offset, barriers[i].offset + barriers[i].size);
    const uint32_t n = u32(count);
    const VkPipelineStageFlags src = acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkPipelineStageFlags dst = acquire ? VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dev_->fn.vkCmdPipelineBarrier(open(), src, dst, 0, 0, nullptr, n, barriers, 0, nullptr);
}

inline void VulkanBackend::external_ownership(Buffer& buffer, size_t offset, size_t bytes, bool acquire) {
    const VkBufferMemoryBarrier mb = external_ownership_barrier(buffer, offset, bytes, acquire);
    if (bytes) external_ownership(&mb, 1, acquire);
}

// A tensor group's sum on this member (VulkanCollective): residual += the `members` sources added in member order, or with `replace` residual = them, the summed values also written to `kept` where given (shaders/sum_rows.comp).
inline void VulkanBackend::sum_rows(Slice residual, const CSlice* sources, size_t members, size_t n, const Slice* kept, bool replace) {
    if (!n) return;
    if (!members || members > kSumSources) throw std::logic_error("vulkan: a sum of no sources or more than the kernel takes");
    const uint32_t pc[3] = {u32(n), u32(members), (kept ? 1u : 0u) | (replace ? 2u : 0u)};
    auto source = [&](size_t k) { return bind(sources[k < members ? k : 0]); };
    dispatch(K_SUM_ROWS, {bind(residual), kept ? bind(*kept) : bind(residual), source(0), source(1), source(2), source(3)}, pc, sizeof(pc), groups(n, 256));
}

// Where share k of a two-shot sum of `n` floats over `members` begins; the last shares are empty where n is smaller than the members.
// Shares are the equal ones rounded up to `align`, the devices' storage-buffer offset alignment in floats, since each is bound at its own offset.
inline size_t share_size(size_t n, size_t members, size_t align) { return ((n + members - 1) / members + align - 1) / align * align; }
inline size_t share_begin(size_t k, size_t n, size_t members, size_t align) { return std::min(n, k * share_size(n, members, align)); }

// A tensor group's ordered sum through per-peer dma-buf inboxes and sync files, alternating two parities (docs/TENSOR-SPLIT.md, section 4.3).
// Each inbox and gather buffer has one importer to avoid serializing unrelated peers.
class VulkanCollective final : public Collective {
public:
    VulkanCollective(const std::vector<VulkanBackend*>& members, size_t rows, size_t width)
        : members_(members), rows_(rows), width_(width), taken_(members.size(), 0), acquires_(members.size()) {
        const size_t W = members.size(), bytes = size_mul(size_mul(rows, width), sizeof(float));
        try {
            threads_ = std::make_unique<core::JobThreads>(W < 3 ? 0 : W - 1);
            for (VulkanBackend* m : members) {
                partial_.push_back(m->alloc(bytes, Memory::device));
                scratch_.push_back(m->alloc(bytes, Memory::device));
            }
            for (int p = 0; p < 2; ++p) {
                inbox_[p].assign(W, std::vector<std::shared_ptr<VulkanBuffer>>(W));
                imported_[p].assign(W, std::vector<BufferPtr>(W));
                for (size_t t = 0; t < W; ++t)
                    for (size_t d = 0; d < W; ++d) {
                        if (d == t) continue;
                        inbox_[p][t][d] = std::make_shared<VulkanBuffer>(members[t]->device(), bytes, -1);
                        imported_[p][t][d] = std::make_shared<VulkanBuffer>(members[d]->device(), bytes, inbox_[p][t][d]->export_fd());
                    }
            }
            for (VulkanBackend* m : members)
                share_align_ = std::max<size_t>(share_align_, ((size_t)m->device()->props.limits.minStorageBufferOffsetAlignment + sizeof(float) - 1) / sizeof(float));
            if (VulkanBackend::two_shot_crossover(W) <= size_mul(rows, width))
                for (int p = 0; p < 2; ++p) {
                    const size_t share = size_mul(share_size(size_mul(rows, width), W, share_align_), sizeof(float));
                    gather_[p].assign(W, std::vector<std::shared_ptr<VulkanBuffer>>(W));
                    gathered_[p].assign(W, std::vector<BufferPtr>(W));
                    for (size_t t = 0; t < W; ++t)
                        for (size_t d = 0; d < W; ++d) {
                            if (d == t) continue;
                            gather_[p][t][d] = std::make_shared<VulkanBuffer>(members[t]->device(), share, -1);
                            gathered_[p][t][d] = std::make_shared<VulkanBuffer>(members[d]->device(), share, gather_[p][t][d]->export_fd());
                        }
                }
            make_semaphores();
            if (W > 1) {
                // One queue thread for the group, kept by its members, so every call to their queues is made there in the order recorded.
                submitter_ = members[0]->submitter();
                if (!submitter_) submitter_ = std::make_shared<VulkanSubmitter>();
                for (VulkanBackend* m : members) m->submit_through(submitter_);
                submitter_->own(this, [this] { recover(); });
            }
        } catch (...) {
            release();
            throw;
        }
    }
    // The queue thread's calls go first, then the member threads: release drains members they call; the context that holds a collective goes before the model's backends.
    ~VulkanCollective() override {
        if (submitter_) {
            submitter_->disown(this);
            submitter_->drain();
        }
        threads_.reset();
        release();
        // The last collective over the group gives the members their own queue calls back.
        if (submitter_ && submitter_.use_count() == 1 + (long)members_.size())
            for (VulkanBackend* m : members_) m->submit_alone();
    }

    Slice partial(size_t member) override { return {partial_.at(member).get(), 0}; }

    void wait(size_t back) noexcept override {
        const std::vector<Ticket>& sum = back ? before_ : last_;
        for (size_t m = 0; m < sum.size(); ++m) members_[m]->wait(sum[m]);
    }

    void sum_into(const std::vector<Slice>& residual, size_t rows, size_t width) override {
        if (residual.size() != members_.size() || rows > rows_ || width != width_)
            throw std::logic_error("backend: a sum of other members, rows or width than its collective was made for");
        summed(residual, size_mul(rows, width), false);
    }

    // Each member's own partial rows are its residual, which the sum replaces.
    void sum_in_place(size_t rows, size_t width) override {
        const size_t n = size_mul(rows, width);
        if (n > size_mul(rows_, width_)) throw std::logic_error("backend: a sum past the rows its collective was made for");
        std::vector<Slice> own;
        for (size_t m = 0; m < members_.size(); ++m) own.push_back(partial(m));
        summed(own, n, true);
    }

private:
    // A sum that fails part way drains the members and makes the semaphores again, so the collective is as new and the passes beside the failed one go on.
    void summed(const std::vector<Slice>& residual, size_t n, bool replace) {
        if (signal_.empty()) throw std::runtime_error("vulkan: a tensor group's semaphores could not be made again after a failed sum");
        try {
            sum(residual, n, replace);
            before_ = last_;
            last_ = taken_;
        } catch (...) {
            release();
            try { make_semaphores(); } catch (const std::exception&) { release(); }
            throw;
        }
    }

    // Adjacent inbox acquires share one command; all spans are checked before recording it.
    void acquire_inboxes(size_t member, int parity, size_t offset, size_t bytes) {
        VulkanBackend& b = *members_[member];
        size_t count = 0;
        for (size_t k = 0; k < members_.size(); ++k)
            if (k != member) {
                const auto mb = b.external_ownership_barrier(*inbox_[parity][member][k], offset, bytes, true);
                if (bytes) acquires_[count++] = mb;
            }
        b.external_ownership(acquires_.data(), count, true);
    }

    void sum(const std::vector<Slice>& residual, size_t n, bool replace) {
        const size_t W = members_.size();
        const size_t bytes = n * sizeof(float);
        // The parity is the collective's, flipped once a sum whichever pass makes it, so passes may interleave their sums here.
        // A sum's adds are submitted by the next exchange, and two parities are enough only because of that.
        const int p = parity_;
        parity_ ^= 1;
        if (n >= VulkanBackend::two_shot_crossover(W)) return two_shots(residual, n, p, replace);
        for (size_t m = 0; m < W; ++m)
            for (size_t t = 0; t < W; ++t)
                if (t != m) {
                    members_[m]->copy(*imported_[p][t][m], 0, *partial_[m], 0, bytes);
                    members_[m]->external_ownership(*imported_[p][t][m], 0, bytes, false);
                }
        exchange();
        for (size_t m = 0; m < W; ++m) {
            VulkanBackend& b = *members_[m];
            acquire_inboxes(m, p, 0, bytes);
            auto source = [&](size_t k) { return CSlice{k == m ? partial_[m].get() : inbox_[p][m][k].get(), 0}; };
            CSlice sources[VulkanBackend::kSumSources];
            for (size_t k = 0; k < W; ++k) sources[k] = source(k);
            b.sum_rows(residual[m], sources, W, n, nullptr, replace);
        }
    }

    // With `replace` each share is written where it is summed and copied where it is gathered, so every member ends with the sum alone.
    void two_shots(const std::vector<Slice>& residual, size_t n, int p, bool replace) {
        const size_t W = members_.size(), f = sizeof(float);
        auto lo = [&](size_t k) { return share_begin(k, n, W, share_align_); };
        auto len = [&](size_t k) { return lo(k + 1) - lo(k); };
        for (size_t m = 0; m < W; ++m)
            for (size_t t = 0; t < W; ++t)
                if (t != m && len(t)) {
                    members_[m]->copy(*imported_[p][t][m], lo(t) * f, *partial_[m], lo(t) * f, len(t) * f);
                    members_[m]->external_ownership(*imported_[p][t][m], lo(t) * f, len(t) * f, false);
                }
        exchange();
        for (size_t k = 0; k < W; ++k) {
            if (!len(k)) continue;
            VulkanBackend& b = *members_[k];
            acquire_inboxes(k, p, lo(k) * f, len(k) * f);
            auto source = [&](size_t j) { return CSlice{j == k ? partial_[k].get() : inbox_[p][k][j].get(), lo(k)}; };
            const Slice share{scratch_[k].get(), lo(k)};
            CSlice sources[VulkanBackend::kSumSources];
            for (size_t j = 0; j < W; ++j) sources[j] = source(j);
            b.sum_rows(Slice{residual[k].buffer, residual[k].offset + lo(k)}, sources, W, len(k), &share, replace);
            for (size_t t = 0; t < W; ++t)
                if (t != k) {
                    b.copy(*gathered_[p][t][k], 0, *scratch_[k], lo(k) * f, len(k) * f);
                    b.external_ownership(*gathered_[p][t][k], 0, len(k) * f, false);
                }
        }
        exchange();
        for (size_t m = 0; m < W; ++m)
            for (size_t k = 0; k < W; ++k)
                if (k != m && len(k)) {
                    members_[m]->external_ownership(*gather_[p][m][k], 0, len(k) * f, true);
                    if (replace) members_[m]->copy(*residual[m].buffer, (residual[m].offset + lo(k)) * f, *gather_[p][m][k], 0, len(k) * f);
                    else members_[m]->add(Slice{residual[m].buffer, residual[m].offset + lo(k)}, CSlice{gather_[p][m][k].get(), 0}, len(k));
                }
    }

    // Every member's submission ended and its ticket taken on the recording thread, its queue call, export and the peers' imports handed to the group's queue thread.
    // The peers' next submissions name the imported semaphores now; the queue thread imports them before it makes any later call.
    void exchange() {
        const size_t W = members_.size();
        std::vector<VulkanBackend::Submission> work(W);
        if (!submitter_) {
            // One member: its own queue call, nothing to exchange.
            work[0] = members_[0]->prepare({});
            taken_[0] = members_[0]->queued(work[0], work[0].queue(*members_[0]->device()));
            return;
        }
        for (size_t m = 0; m < W; ++m) {
            std::vector<VkSemaphore> signals;
            if (signal_[m]) signals.push_back(signal_[m]);
            work[m] = members_[m]->prepare(signals);
            taken_[m] = members_[m]->queued(work[m], VK_SUCCESS);
        }
        for (size_t m = 0; m < W; ++m)
            for (size_t t = 0; t < W; ++t)
                if (t != m) members_[t]->wait_on(wait_[t][m]);
        submitter_->run({[this, work] { exchange_calls(work); },
                         [this, work] {
                             for (size_t m = 0; m < work.size(); ++m) members_[m]->made(work[m].ticket, false);
                         }});
    }

    // On the queue thread: queue and export concurrently from three members on, then import each peer's fence.
    // Member threads touch only their queue and semaphore exports; all are joined before return or throw.
    void exchange_calls(const std::vector<VulkanBackend::Submission>& work) {
        const size_t W = members_.size();
        struct Files {
            std::vector<int> fd;
            ~Files() {
#if !defined(_WIN32)
                for (int f : fd)
                    if (f >= 0) close(f);
#endif
            }
        } files{std::vector<int>(W * W, -1)};
        std::vector<VkResult> result(W, VK_NOT_READY);
        threads_->run(W, [&](size_t m) {
            const Device& from = *members_[m]->device();
            result[m] = work[m].queue(from);
            if (result[m] != VK_SUCCESS) return;
            members_[m]->made(work[m].ticket, true);
            const size_t first = m ? 0 : 1;
            VkSemaphoreGetFdInfoKHR gi{};
            gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
            gi.semaphore = signal_[m];
            gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            int& fd = files.fd[m * W + first];
            check(from.get_semaphore_fd(from.device, &gi, &fd), "vkGetSemaphoreFdKHR");
            // A sync file is an immutable fence; each peer takes its own descriptor, including -1 for a completed signal.
            for (size_t t = 0; t < W; ++t) {
                if (t == m || t == first || fd < 0) continue;
#if !defined(_WIN32)
                const int copy = fcntl(fd, F_DUPFD_CLOEXEC, 3);
                if (copy < 0) throw std::runtime_error("vulkan: duplicating a sync file failed");
                files.fd[m * W + t] = copy;
#else
                throw std::runtime_error("vulkan: sync files require POSIX descriptors");
#endif
            }
        });
        for (size_t m = 0; m < W; ++m) check(result[m], "vkQueueSubmit");
        for (size_t m = 0; m < W; ++m) {
            for (size_t t = 0; t < W; ++t) {
                if (t == m) continue;
                const Device& to = *members_[t]->device();
                VkImportSemaphoreFdInfoKHR ii{};
                ii.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                ii.semaphore = wait_[t][m];
                ii.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                ii.fd = files.fd[m * W + t];
                check(to.import_semaphore_fd(to.device, &ii), "vkImportSemaphoreFdKHR");
                files.fd[m * W + t] = -1;
            }
        }
    }

    void make_semaphores() {
        const size_t W = members_.size();
        signal_.assign(W, VK_NULL_HANDLE);
        wait_.assign(W, std::vector<VkSemaphore>(W, VK_NULL_HANDLE));
        for (size_t m = 0; m < W; ++m) {
            if (W > 1) signal_[m] = semaphore(*members_[m]->device(), true);
            for (size_t t = 0; t < W; ++t) {
                if (t == m) continue;
                wait_[t][m] = semaphore(*members_[t]->device(), false);
            }
        }
    }
    static VkSemaphore semaphore(const Device& d, bool exported) {
        VkExportSemaphoreCreateInfo ex{};
        ex.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        ex.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        si.pNext = exported ? &ex : nullptr;
        VkSemaphore s = VK_NULL_HANDLE;
        check(d.fn.vkCreateSemaphore(d.device, &si, nullptr, &s), "vkCreateSemaphore");
        return s;
    }
    // On the recording thread given a failed queue call: each member's given-up tickets settled (`give_up`) and the semaphores made again.
    void recover() noexcept {
        for (VulkanBackend* m : members_) m->give_up();
        destroy_semaphores();
        try { make_semaphores(); } catch (const std::exception&) { destroy_semaphores(); }
    }
    // Every member drained first, so no submission still waits on, signals or reads what goes.
    void release() noexcept {
        for (VulkanBackend* m : members_) m->sync();
        if (submitter_) submitter_->drain();
        destroy_semaphores();
    }
    void destroy_semaphores() noexcept {
        for (size_t a = 0; a < signal_.size(); ++a)
            if (signal_[a]) members_[a]->device()->fn.vkDestroySemaphore(members_[a]->device()->device, signal_[a], nullptr);
        for (size_t a = 0; a < wait_.size(); ++a)
            for (size_t b = 0; b < wait_[a].size(); ++b)
                if (wait_[a][b]) members_[a]->device()->fn.vkDestroySemaphore(members_[a]->device()->device, wait_[a][b], nullptr);
        signal_.clear();
        wait_.clear();
    }

    std::vector<VulkanBackend*> members_;
    size_t rows_, width_;
    int parity_ = 0;
    std::vector<Ticket> taken_, last_, before_;   // per member, its submission of the last exchange, of the last sum and of the sum before it
    std::vector<VkBufferMemoryBarrier> acquires_;   // recording-thread scratch, sized once for every peer
    std::unique_ptr<core::JobThreads> threads_;   // a thread a member but the first, asleep between exchanges, from three members on (docs/STATUS.md, a group's members submit side by side)
    std::vector<BufferPtr> partial_, scratch_;
    std::vector<std::vector<std::shared_ptr<VulkanBuffer>>> inbox_[2];   // [parity][owner][sender]: the owner's inbox for that sender's partial rows
    std::vector<std::vector<BufferPtr>> imported_[2];                    // [parity][owner][sender]: that inbox on the sender, its only importer
    size_t share_align_ = 1;   // the members' largest storage-buffer offset alignment, in floats
    std::vector<std::vector<std::shared_ptr<VulkanBuffer>>> gather_[2];  // [parity][owner][sender]: the owner's gather rows for that sender's share, where a sum may take two shots
    std::vector<std::vector<BufferPtr>> gathered_[2];                    // [parity][owner][sender]: those gather rows on the sender, their only importer
    std::vector<VkSemaphore> signal_;                  // one binary signal per source, exported once for every peer
    std::vector<std::vector<VkSemaphore>> wait_;        // [member][peer]: what member waits on from the peer
    std::shared_ptr<VulkanSubmitter> submitter_;        // the group's queue thread, from two members on
};

// A collective over Vulkan devices of one profile, each opened once, that share memory as dma-buf and semaphores as sync files, under any PCI root complexes; none where a device lacks them, as on Windows.
std::unique_ptr<Collective> VulkanBackend::join(const std::vector<Backend*>& members, size_t rows, size_t width) {
    if (members.empty() || members[0] != this) throw std::logic_error("backend: a collective is joined by its first member");
    if (members.size() > kSumSources) throw std::runtime_error("backend: a Vulkan collective of more than " + std::to_string(kSumSources) + " devices");
    std::vector<VulkanBackend*> group;
    for (Backend* b : members) {
        auto* v = dynamic_cast<VulkanBackend*>(b);
        if (!v) throw std::runtime_error("backend: a Vulkan collective over a backend that is not Vulkan");
        if (v->dev_->caps.device != dev_->caps.device || v->dev_->caps.driver != dev_->caps.driver)
            throw std::runtime_error("backend: a Vulkan collective over devices of different kinds");
        for (VulkanBackend* other : group)
            if (other->dev_->physical == v->dev_->physical) throw std::runtime_error("backend: a Vulkan collective over one device twice");
        if (!v->dev_->exchange) return nullptr;
        group.push_back(v);
    }
    // Members may sit under different PCI root complexes, which cost the same on the machines measured (docs/TENSOR-SPLIT.md, section 4.5); a group is refused only where its memory or semaphores cannot be shared.
    try {
        return std::make_unique<VulkanCollective>(group, rows, width);
    } catch (const std::exception& e) {
        std::string names;
        for (const VulkanBackend* v : group) names += (names.empty() ? "" : ", ") + v->dev_->caps.device + (v->dev_->pci_root.empty() ? "" : " under " + v->dev_->pci_root);
        throw std::runtime_error("backend: a tensor group of " + names + " cannot share its memory and semaphores: " + e.what());
    }
}

} // namespace
} // namespace backend
