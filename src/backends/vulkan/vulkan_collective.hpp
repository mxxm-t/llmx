#pragma once
// A tensor group's collective over Vulkan devices and VulkanBackend's part in it; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_device.hpp"
#include "core/job_threads.hpp"

namespace backend {
namespace {

// Per peer, an inbox for each of two parities of each of two lanes (Collective::send); lane 0's also take every sum made in one call.
constexpr size_t kInboxes = 4;

// Partial, scratch and per-peer inbox rows for both parities of both lanes; two-shot sums also reserve each peer's gather share.
// Imports consume no memory on the importing device.
inline size_t VulkanBackend::collective_bytes_per_row(size_t members, size_t width) const {
    if (!dev_->exchange || !members) return 0;
    const size_t rows = size_add(size_add(2, size_mul(kInboxes, members - 1)), two_shot_crossover(members) != SIZE_MAX ? 2 : 0);
    return size_mul(rows, size_mul(width, sizeof(float)));
}

// The measured crossover for two-shot sums on this transport (docs/STATUS.md, a group's large sums in two shots).
// Two members keep broadcast because two shots move the same bytes with another wait round.
inline size_t VulkanBackend::two_shot_crossover(size_t members) {
    return members < 3 ? SIZE_MAX : 327680;
}

// The largest sum the tagged route takes (docs/STATUS.md, Vulkan collective route); zero until the size switch is frozen, so the sync-file path stays the default.
inline size_t VulkanBackend::tagged_crossover(size_t) { return 0; }

// One tagged sum's sender on this member (flag_push mode 16 | 8, lag 1) and the gate after it (mode 8, lag 0), which holds the stream until its own receiver acked the epoch.
// The receiver on the other queue wrote `rows` in place, so the main stream forgets it was written here.
inline void VulkanBackend::tagged_push(Slice rows, Buffer& control, const VkDescriptorBufferInfo* peers, size_t n, uint32_t epoch, size_t self, size_t members, uint32_t gen,
                                       uint32_t slots, uint32_t cap) {
    constexpr uint32_t C = 256;
    const uint32_t G = u32(((n + 2) / 3 + C - 1) / C);
    const VkDescriptorBufferInfo src = bind(rows), ctl = bind(Slice{&control, 0});
    uint32_t pc[13] = {u32(n), epoch, u32(self), u32(members), G, C, G, 0, 1, cap, 16u | 8u, gen, slots};
    static_assert(kMaxTensorWidth == 16 && 2 + kMaxTensorWidth <= kMaxBuffers, "a tagged region a member below");
    // On RADV coherent bindings load from L2, so after compute writers execution order suffices and a write-back is all the gate needs; the Vulkan model promises neither, so other drivers take full barriers (docs/src/backends-vulkan.md).
    // A transfer writer's data may not be in L2, so it takes the full barrier.
    const bool weak = dev_->radv;
    if (weak && since_ == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) compute_barrier(open(), 0);
    else if (since_ || owed_) {
        VkCommandBuffer cmd = open();
        barrier(cmd);
        settle_all(cmd);
    }
    bare_ = true;
    struct Clear {
        bool& bare;
        ~Clear() { bare = false; }
    } clear{bare_};
    for (int pass = 0; pass < 2; ++pass) {
        dispatch(K_FLAG_PUSH, {src, ctl, peers[0], peers[1], peers[2], peers[3], peers[4], peers[5], peers[6], peers[7], peers[8], peers[9], peers[10], peers[11], peers[12],
                               peers[13], peers[14], peers[15]},
                 pc, sizeof(pc), G);
        // The sender's stores to the peers' inboxes stay in its L2 until a write-back, which the gate waits for.
        if (pass == 0 && weak) compute_barrier(open(), VK_ACCESS_SHADER_WRITE_BIT);
        // Under matched barriers barrier() only owes one and the bare gate settles nothing, so the barrier is settled here.
        if (pass == 0 && !weak) {
            VkCommandBuffer cmd = open();
            barrier(cmd);
            settle(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        pc[8] = 0;
        pc[10] = 8u;
    }
    // What follows reads the rows the receiver wrote and the writes whose barrier waited, so the full barrier, or the one owed, follows the gate.
    barrier(open());
    forget_written(src.buffer, 0, VK_WHOLE_SIZE);
}

// A receiver run recorded on a command buffer of the receive queue's family (shaders/flag_recv.comp).
inline void VulkanBackend::record_receiver(VkCommandBuffer cmd, const VkDescriptorBufferInfo* buffers, const uint32_t* push, size_t push_words, uint32_t groups) {
    bind_kernel(cmd, K_FLAG_RECV, kernel(K_FLAG_RECV), buffers, push, push_words * sizeof(uint32_t));
    dev_->fn.vkCmdDispatch(cmd, groups, 1, 1);
}

// Preserve a copied peer range across the collective's semaphore exchange (docs/TENSOR-SPLIT.md, section 4.3).
inline VkBufferMemoryBarrier VulkanBackend::external_ownership_barrier(Buffer& buffer, size_t offset, size_t bytes, bool acquire) {
    VulkanBuffer& b = as_vulkan(buffer);
    span(b, offset, bytes);
    VkBufferMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    mb.srcAccessMask = acquire ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = acquire ? VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT : 0;
    // A buffer concurrent over two families (Device::share) names no family of its own in a transfer.
    const uint32_t own = dev_->receive_family == UINT32_MAX ? dev_->queue_family : VK_QUEUE_FAMILY_IGNORED;
    mb.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_FOREIGN_EXT : own;
    mb.dstQueueFamilyIndex = acquire ? own : VK_QUEUE_FAMILY_FOREIGN_EXT;
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
    settle_all(open());
    dev_->fn.vkCmdPipelineBarrier(open(), src, dst, 0, 0, nullptr, n, barriers, 0, nullptr);
}

inline void VulkanBackend::external_ownership(Buffer& buffer, size_t offset, size_t bytes, bool acquire) {
    const VkBufferMemoryBarrier mb = external_ownership_barrier(buffer, offset, bytes, acquire);
    if (bytes) external_ownership(&mb, 1, acquire);
}

// A tensor group's sum on this member (VulkanCollective): residual += the `members` sources added in member order, or with `replace` residual = them, the summed values also written to `kept` where given (shaders/sum_rows.comp).
inline void VulkanBackend::sum_rows(Slice residual, const CSlice* sources, size_t members, size_t n, const Slice* kept, bool replace) {
    if (!n) return;
    if (!members || members > kMaxTensorWidth) throw std::logic_error("vulkan: a sum of no sources or more than the kernel takes");
    const uint32_t pc[3] = {u32(n), u32(members), (kept ? 1u : 0u) | (replace ? 2u : 0u)};
    auto source = [&](size_t k) { return bind(sources[k < members ? k : 0]); };
    static_assert(kMaxTensorWidth == 16, "a source a member below");
    dispatch(K_SUM_ROWS, {bind(residual), kept ? bind(*kept) : bind(residual), source(0), source(1), source(2), source(3), source(4), source(5), source(6), source(7),
                          source(8), source(9), source(10), source(11), source(12), source(13), source(14), source(15)},
             pc, sizeof(pc), groups(n, 256));
}

// Where share k of a two-shot sum of `n` floats over `members` begins; the last shares are empty where n is smaller than the members.
// Shares are the equal ones rounded up to `align`, the devices' storage-buffer offset alignment in floats, since each is bound at its own offset.
inline size_t share_size(size_t n, size_t members, size_t align) { return ((n + members - 1) / members + align - 1) / align * align; }
inline size_t share_begin(size_t k, size_t n, size_t members, size_t align) { return std::min(n, k * share_size(n, members, align)); }

// A tensor group's ordered sum through per-peer dma-buf inboxes and timeline semaphores, alternating two parities (docs/TENSOR-SPLIT.md, section 4.3).
// Each inbox and gather buffer has one importer to avoid serializing unrelated peers.
class VulkanCollective final : public Collective {
public:
    // Sums in place of at most `tagged` floats take the tagged route where every member has what it needs; zero leaves it off.
    VulkanCollective(const std::vector<VulkanBackend*>& members, size_t rows, size_t width, size_t tagged)
        : members_(members), rows_(rows), width_(width), taken_(members.size(), 0), acquires_(members.size()) {
        const size_t W = members.size(), bytes = size_mul(size_mul(rows, width), sizeof(float));
        try {
            for (VulkanBackend* m : members) {
                partial_.push_back(m->alloc(bytes, Memory::device));
                scratch_.push_back(m->alloc(bytes, Memory::device));
            }
            for (size_t p = 0; p < kInboxes; ++p) {
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
            beside_ = W > 1 && std::all_of(members.begin(), members.end(), [](VulkanBackend* m) { return m->copies_beside(); });
            copied_[0].assign(W, 0);
            copied_[1].assign(W, 0);
            if (W > 1) {
                // One queue thread for the group, kept by its members, so every call to their queues is made there in the order recorded; a member of a live group brings its own.
                for (VulkanBackend* m : members)
                    if (m->submitter() && m->submitter() != submitter_) {
                        if (submitter_) throw std::runtime_error("vulkan: a tensor group over devices of two groups' queue threads");
                        submitter_ = m->submitter();
                    }
                if (!submitter_) submitter_ = std::make_shared<VulkanSubmitter>();
                for (VulkanBackend* m : members) {
                    m->submit_through(submitter_);
                    ++routed_;
                }
                submitter_->own(this, [this] { recover(); });
                if (tagged) tag(tagged);
            }
        } catch (...) {
            dismantle();
            throw;
        }
    }
    ~VulkanCollective() override { dismantle(); }

    Slice partial(size_t member) override { return {partial_.at(member).get(), 0}; }

    // A lane sends its whole part to every peer, so it copies beside only where a sum of its size made in one call would take one shot.
    bool copies_beside(size_t floats) const override { return beside_ && floats < VulkanBackend::two_shot_crossover(members_.size()); }

    // An open batch of tagged sums is closed first; an abort its receivers report turns the route off and is thrown by the next sum, since a wait cannot throw.
    // The close does not give out a queue-thread failure, which the next hand-over then throws.
    void wait(size_t back) noexcept override {
        if (!runs_.empty()) {
            try { close_open(*submitter_, nullptr, false); } catch (...) {}
        }
        const std::vector<Ticket>& sum = back ? before_ : last_;
        for (size_t m = 0; m < sum.size(); ++m) members_[m]->wait(sum[m]);
        if (received_) received();
    }

    void sum_into(const std::vector<Slice>& residual, size_t rows, size_t width) override {
        aborted();
        if (residual.size() != members_.size() || rows > rows_ || width != width_)
            throw std::logic_error("backend: a sum of other members, rows or width than its collective was made for");
        summed(residual, size_mul(rows, width), false);
    }

    // Each member's own partial rows are its residual, which the sum replaces.
    void sum_in_place(size_t rows, size_t width) override {
        const size_t n = size_mul(rows, width);
        if (n > size_mul(rows_, width_)) throw std::logic_error("backend: a sum past the rows its collective was made for");
        aborted();
        if (n && n <= tagged_limit_ && epoch_ < 0x7fff0000u) {
            close_behind();
            if (n <= tagged_limit_) return tagged_sum(n);
        }
        std::vector<Slice> own;
        for (size_t m = 0; m < members_.size(); ++m) own.push_back(partial(m));
        summed(own, n, true);
    }

    void send(size_t lane, size_t base, size_t rows, size_t width) override {
        const size_t W = members_.size(), at = size_mul(size_mul(base, width), sizeof(float)), bytes = size_mul(size_mul(rows, width), sizeof(float));
        if (lane > 1 || size_add(base, rows) > rows_ || width > width_ || sent_[lane] >= 0) throw std::logic_error("backend: a sum of a lane, rows or width its collective was not made for");
        if (signal_.empty()) throw std::runtime_error("vulkan: a tensor group's semaphores could not be made again after a failed sum");
        const int p = int(2 * lane) + lane_parity_[lane];
        lane_parity_[lane] ^= 1;
        try {
            for (size_t m = 0; m < W; ++m)
                for (size_t t = 0; t < W; ++t)
                    if (t != m && beside_) {
                        members_[m]->copy_beside(*imported_[p][t][m], at, *partial_[m], at, bytes);
                    } else if (t != m) {
                        members_[m]->copy(*imported_[p][t][m], at, *partial_[m], at, bytes);
                        members_[m]->external_ownership(*imported_[p][t][m], at, bytes, false);
                    }
            sent_count_[lane] = exchange(lane, beside_);
        } catch (...) {
            failed();
            throw;
        }
        sent_[lane] = p;
    }

    void finish(size_t lane, size_t base, size_t rows, size_t width, const std::vector<Slice>* residual) override {
        const size_t W = members_.size(), n = size_mul(rows, width), at = size_mul(base, width);
        if (lane > 1 || sent_[lane] < 0 || (residual && residual->size() != W)) throw std::logic_error("backend: a sum finished that was not sent, or of other members");
        const int p = sent_[lane];
        sent_[lane] = -1;
        try {
            if (beside_) {
                wait_peers(sent_count_[lane]);
                for (size_t m = 0; m < W; ++m) members_[m]->wait_copies(copied_[lane][m]);
            }
            for (size_t m = 0; m < W; ++m) {
                VulkanBackend& b = *members_[m];
                acquire_inboxes(m, p, at * sizeof(float), n * sizeof(float));
                CSlice sources[kMaxTensorWidth];
                for (size_t k = 0; k < W; ++k) sources[k] = CSlice{k == m ? partial_[m].get() : inbox_[p][m][k].get(), at};
                b.sum_rows(residual ? (*residual)[m] : Slice{partial_[m].get(), at}, sources, W, n, nullptr, !residual);
            }
            before_ = last_;
            last_ = taken_;
        } catch (...) {
            failed();
            throw;
        }
    }

    // A lane sent and not finished leaves the collective as a failed sum does.
    void drop() noexcept override {
        if (sent_[0] >= 0 || sent_[1] >= 0) failed();
    }

private:
    friend struct VulkanSumTest;
    // An abort a wait read, thrown once by the next sum, which with every later one goes by sync files.
    void aborted() {
        if (!aborted_) return;
        aborted_ = false;
        throw std::runtime_error("vulkan: a tensor group's tagged sum aborted; its sums go by sync files from now on");
    }
    static constexpr size_t kReceivers = 4;   // receiver command buffers a member, reused once the receive timeline passed their batch

    // The tagged route's state, made where every member has a receive queue, a clock its shaders read and uncached memory, for sums of at most `limit` floats.
    // Each owner's tagged rows for one sender, 8 rows of `slots_` 16-byte slots, are exported by the owner and imported by that sender alone, as the inboxes are.
    void tag(size_t limit) {
        const size_t W = members_.size();
        if (!pools_.empty()) return;
        for (VulkanBackend* m : members_) {
            const Device& d = *m->device();
            if (!d.receive_queue || !d.device_clock || !d.uncached_memory || !(d.timestamp_ns > 0)) return;
        }
        limit = std::min(limit, size_mul(rows_, width_));
        rsent_.assign(W, 0);
        slots_ = uint32_t((limit + 2) / 3);
        const uint32_t G = (slots_ + 255) / 256;
        cap_ = (uint32_t)std::min<double>(4.0e9, 1.0e9 / members_[0]->device()->timestamp_ns);
        const size_t tag_bytes = size_t(8) * slots_ * 16, ctl_bytes = size_t(16) * (G + 2) * sizeof(uint32_t);
        tags_.assign(W, std::vector<std::shared_ptr<VulkanBuffer>>(W));
        tagged_in_.assign(W, std::vector<BufferPtr>(W));
        const std::vector<uint8_t> zeros(std::max(tag_bytes, ctl_bytes), 0);
        for (size_t t = 0; t < W; ++t) {
            ctl_.push_back(std::make_shared<VulkanBuffer>(members_[t]->device(), ctl_bytes, -1, true));
            report_.push_back(std::make_shared<VulkanBuffer>(members_[t]->device(), 4 * sizeof(uint32_t), true));
            members_[t]->write(*ctl_[t], 0, zeros.data(), ctl_bytes);
            for (size_t d = 0; d < W; ++d) {
                if (d == t) continue;
                tags_[t][d] = std::make_shared<VulkanBuffer>(members_[t]->device(), tag_bytes, -1, true);
                members_[t]->write(*tags_[t][d], 0, zeros.data(), tag_bytes);
                tagged_in_[t][d] = std::make_shared<VulkanBuffer>(members_[d]->device(), tag_bytes, tags_[t][d]->export_fd(), true);
            }
            const Device& dv = *members_[t]->device();
            VkCommandPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = dv.receive_family;
            pools_.push_back(VK_NULL_HANDLE);
            check(dv.fn.vkCreateCommandPool(dv.device, &pi, nullptr, &pools_.back()), "vkCreateCommandPool");
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = pools_.back();
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = kReceivers;
            rcmds_.emplace_back(kReceivers);
            check(dv.fn.vkAllocateCommandBuffers(dv.device, &ai, rcmds_.back().data()), "vkAllocateCommandBuffers");
            rtimeline_.push_back(semaphore(dv));
        }
        for (VulkanBackend* m : members_) m->sync();
        tagged_limit_ = limit;
    }

    // A tagged sum after an open collective that shares a member and opened later closes the batch first: a device's receive queue takes the open collectives' receivers one collective after another, in open order.
    void close_behind() {
        const std::vector<VulkanCollective*>& open = submitter_->open_batches;
        const auto self = std::find(open.begin(), open.end(), this);
        if (self != open.end() && std::any_of(self + 1, open.end(), [this](const VulkanCollective* o) { return shares_member(*o); })) close_open(*submitter_, nullptr, true);
    }

    // One sum in place by the tagged route, recorded on every member and left in the open batch; consecutive sums of one n extend one receiver run.
    void tagged_sum(size_t n) {
        const size_t W = members_.size();
        const uint32_t e = ++epoch_;
        if (runs_.empty() || runs_.back().n != n) runs_.push_back({n, e - 1, 0});
        ++runs_.back().epochs;
        for (VulkanBackend* m : members_) m->defer_chunk = true;
        for (size_t m = 0; m < W; ++m) {
            VkDescriptorBufferInfo peers[kMaxTensorWidth];
            for (size_t k = 0; k < kMaxTensorWidth; ++k)
                peers[k] = k < W && k != m ? VkDescriptorBufferInfo{static_cast<VulkanBuffer&>(*tagged_in_[k][m]).handle(), 0, VK_WHOLE_SIZE} : VkDescriptorBufferInfo{ctl_[m]->handle(), 0, VK_WHOLE_SIZE};
            members_[m]->tagged_push(partial(m), *ctl_[m], peers, n, e, m, W, gen_, slots_, cap_);
        }
        for (VulkanBackend* m : members_) m->defer_chunk = false;
        std::vector<VulkanCollective*>& open = submitter_->open_batches;
        if (std::find(open.begin(), open.end(), this) == open.end()) open.push_back(this);
        ++tagged_sums_;
    }

public:
    // Every open batch of the group closed in one job of its queue thread: each collective's receivers recorded, each member of any of them (and `caller`) prepared once, all their submissions made, then every receiver run on its receive queue.
    // Collectives sharing a member share its stream, and no semaphore joins the queues, so all close together and the kernels wait on acks and gates (docs/STATUS.md); returns `caller`'s ticket.
    static Ticket close_open(VulkanSubmitter& s, VulkanBackend* caller, bool report) {
        const std::vector<VulkanCollective*> open = std::move(s.open_batches);
        s.open_batches.clear();
        std::vector<uint64_t> values(open.size());
        for (size_t c = 0; c < open.size(); ++c) {
            VulkanCollective& g = *open[c];
            values[c] = g.received_ + 1;
            try {
                for (size_t t = 0; t < g.members_.size(); ++t) g.record_receivers(t, values[c] % kReceivers);
            } catch (...) {
                for (VulkanCollective* o : open) o->drop_batch();
                throw;
            }
        }
        std::vector<VulkanBackend*> backs;
        if (caller) backs.push_back(caller);
        for (VulkanCollective* g : open)
            for (VulkanBackend* m : g->members_)
                if (std::find(backs.begin(), backs.end(), m) == backs.end()) backs.push_back(m);
        std::vector<VulkanBackend::Submission> work(backs.size());
        std::vector<Ticket> tickets(backs.size());
        for (size_t i = 0; i < backs.size(); ++i) {
            try {
                work[i] = backs[i]->prepare({});
            } catch (...) {
                for (size_t j = 0; j < i; ++j) backs[j]->made(work[j].ticket, false);
                for (VulkanCollective* o : open) o->drop_batch();
                throw;
            }
            tickets[i] = backs[i]->queued(work[i], VK_SUCCESS);
        }
        for (size_t c = 0; c < open.size(); ++c) {
            VulkanCollective& g = *open[c];
            g.received_ = values[c];
            g.rqueued_[values[c] % kReceivers] = values[c];
            for (size_t m = 0; m < g.members_.size(); ++m) g.taken_[m] = tickets[std::find(backs.begin(), backs.end(), g.members_[m]) - backs.begin()];
            g.before_ = g.last_ = g.taken_;
            g.runs_.clear();
            ++g.gen_;
        }
        // Once the queue calls begin they settle their own tickets, so a settle after a call that threw gives up only what never reached a queue.
        const auto called = std::make_shared<bool>(false);
        s.run({[sp = &s, called, open, values, backs, work] {
                   queue_calls(sp->helpers(backs.size()), backs, work, *called);
                   for (size_t c = 0; c < open.size(); ++c) open[c]->submit_receivers(values[c]);
               },
               [called, open, values, backs, work] {
                   if (!*called)
                       for (size_t i = 0; i < backs.size(); ++i) backs[i]->made(work[i].ticket, false);
                   for (size_t c = 0; c < open.size(); ++c) open[c]->give_up_receivers(values[c]);
               }},
              report);
        return caller ? tickets[0] : 0;
    }

private:
    // An open batch whose receivers could not be recorded is dropped and the route goes off: its senders and gates abort at the cap.
    void drop_batch() noexcept {
        runs_.clear();
        ++gen_;
        tagged_limit_ = 0;
    }

    // On the queue thread: each member's receivers of batch `value` on its receive queue, which signal its receive timeline at that value.
    void submit_receivers(uint64_t value) {
        for (size_t t = 0; t < members_.size(); ++t) {
            const Device& d = *members_[t]->device();
            VkTimelineSemaphoreSubmitInfo tsi{};
            tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            tsi.signalSemaphoreValueCount = 1;
            tsi.pSignalSemaphoreValues = &value;
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.pNext = &tsi;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &rcmds_[t][value % kReceivers];
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &rtimeline_[t];
            if (queue_probe_) check(queue_probe_(*members_[t], true), "vkQueueSubmit (receivers)");
            check(d.fn.vkQueueSubmit(d.receive_queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit (receivers)");
            rsent_[t] = value;
        }
    }

    // Receivers given up still reach their value, so no wait on them hangs; the route goes off.
    // Those already on their receive queue signal it themselves, and a host signal beside a pending one would be invalid.
    void give_up_receivers(uint64_t value) noexcept {
        for (size_t t = 0; t < members_.size(); ++t) {
            if (rsent_[t] >= value) continue;
            const Device& d = *members_[t]->device();
            VkSemaphoreSignalInfo si{};
            ++host_signals_;
            si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
            si.semaphore = rtimeline_[t];
            si.value = value;
            d.fn.vkSignalSemaphore(d.device, &si);
        }
        aborted_ = true;
        tagged_limit_ = 0;
    }

    // Owner t's receivers of the open batch in ring slot `slot`, once the slot's last batch has finished, then its abort words copied where the host reads them.
    void record_receivers(size_t t, size_t slot) {
        const Device& d = *members_[t]->device();
        wait_received(t, rqueued_[slot]);
        VkCommandBuffer cmd = rcmds_[t][slot];
        check(d.fn.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(d.fn.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        const size_t W = members_.size();
        const VkDescriptorBufferInfo own{static_cast<VulkanBuffer&>(*partial_[t]).handle(), 0, VK_WHOLE_SIZE}, ctl{ctl_[t]->handle(), 0, VK_WHOLE_SIZE};
        std::vector<VkDescriptorBufferInfo> b(5 + kMaxTensorWidth, ctl);
        b[0] = b[1] = b[2] = own;
        for (size_t k = 0; k < W; ++k)
            if (k != t) b[5 + k] = VkDescriptorBufferInfo{tags_[t][k]->handle(), 0, VK_WHOLE_SIZE};
        for (const Run& r : runs_) {
            const uint32_t G = uint32_t(((r.n + 2) / 3 + 255) / 256);
            const uint32_t pc[15] = {uint32_t(r.n), r.base, r.epochs, uint32_t(t), uint32_t(W), G, 256, G, 0, 0, G, cap_, 8u | 2u, gen_, slots_};
            // Runs one after another: each rebases the acks at its start, which must not overtake the run before it.
            if (&r != &runs_.front()) {
                VkMemoryBarrier rb{};
                rb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                rb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                rb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                d.fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rb, 0, nullptr, 0, nullptr);
            }
            members_[t]->record_receiver(cmd, b.data(), pc, 15, G);
        }
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        d.fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        const VkBufferCopy words{0, 0, 4 * sizeof(uint32_t)};
        d.fn.vkCmdCopyBuffer(cmd, ctl_[t]->handle(), report_[t]->handle(), 1, &words);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        d.fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        check(d.fn.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    }

    bool shares_member(const VulkanCollective& o) const {
        for (VulkanBackend* m : members_)
            if (std::find(o.members_.begin(), o.members_.end(), m) != o.members_.end()) return true;
        return false;
    }

    bool wait_received(size_t t, uint64_t value) noexcept {
        if (!value) return true;
        const Device& d = *members_[t]->device();
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &rtimeline_[t];
        wi.pValues = &value;
        return d.fn.vkWaitSemaphores(d.device, &wi, UINT64_MAX) == VK_SUCCESS;
    }

    // After the last closed batch's receivers: a batch any member's control word 0 names as aborted turns the route off, and the next tagged sum throws.
    void received() noexcept {
        for (size_t t = 0; t < members_.size(); ++t) {
            const uint32_t word = wait_received(t, received_) ? static_cast<const uint32_t*>(report_[t]->mapped())[0] : UINT32_MAX;
            if (word > reported_) {
                reported_ = word;
                aborted_ = true;
                tagged_limit_ = 0;
            }
        }
    }
    // A sum that fails part way drains the members and makes the semaphores again, so the collective is as new and the passes beside the failed one go on.
    void summed(const std::vector<Slice>& residual, size_t n, bool replace) {
        if (signal_.empty()) throw std::runtime_error("vulkan: a tensor group's semaphores could not be made again after a failed sum");
        try {
            sum(residual, n, replace);
            before_ = last_;
            last_ = taken_;
        } catch (...) {
            failed();
            throw;
        }
    }
    void failed() noexcept {
        release();
        for (VulkanBackend* m : members_) m->copies_abandon();
        sent_[0] = sent_[1] = -1;
        try { make_semaphores(); } catch (const std::exception&) { release(); }
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
        // The parity is lane 0's, flipped once a sum whichever pass makes it, and two parities are enough because a sum's adds are submitted by the next exchange.
        // A lane's peers wait for a count it signals from the copy queue, so no main-queue signal may overtake it before the lane's finish.
        if (sent_[0] >= 0 || sent_[1] >= 0) throw std::logic_error("vulkan: a sum made in one call while a lane's sum is sent and not finished");
        const int p = lane_parity_[0];
        lane_parity_[0] ^= 1;
        if (n >= VulkanBackend::two_shot_crossover(W)) return two_shots(residual, n, p, replace);
        for (size_t m = 0; m < W; ++m)
            for (size_t t = 0; t < W; ++t)
                if (t != m) {
                    members_[m]->copy(*imported_[p][t][m], 0, *partial_[m], 0, bytes);
                    members_[m]->external_ownership(*imported_[p][t][m], 0, bytes, false);
                }
        exchange(0, false);
        for (size_t m = 0; m < W; ++m) {
            VulkanBackend& b = *members_[m];
            acquire_inboxes(m, p, 0, bytes);
            auto source = [&](size_t k) { return CSlice{k == m ? partial_[m].get() : inbox_[p][m][k].get(), 0}; };
            CSlice sources[kMaxTensorWidth];
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
        exchange(0, false);
        for (size_t k = 0; k < W; ++k) {
            if (!len(k)) continue;
            VulkanBackend& b = *members_[k];
            acquire_inboxes(k, p, lo(k) * f, len(k) * f);
            auto source = [&](size_t j) { return CSlice{j == k ? partial_[k].get() : inbox_[p][k][j].get(), lo(k)}; };
            const Slice share{scratch_[k].get(), lo(k)};
            CSlice sources[kMaxTensorWidth];
            for (size_t j = 0; j < W; ++j) sources[j] = source(j);
            b.sum_rows(Slice{residual[k].buffer, residual[k].offset + lo(k)}, sources, W, len(k), &share, replace);
            for (size_t t = 0; t < W; ++t)
                if (t != k) {
                    b.copy(*gathered_[p][t][k], 0, *scratch_[k], lo(k) * f, len(k) * f);
                    b.external_ownership(*gathered_[p][t][k], 0, len(k) * f, false);
                }
        }
        exchange(0, false);
        for (size_t m = 0; m < W; ++m)
            for (size_t k = 0; k < W; ++k)
                if (k != m && len(k)) {
                    members_[m]->external_ownership(*gather_[p][m][k], 0, len(k) * f, true);
                    if (replace) members_[m]->copy(*residual[m].buffer, (residual[m].offset + lo(k)) * f, *gather_[p][m][k], 0, len(k) * f);
                    else members_[m]->add(Slice{residual[m].buffer, residual[m].offset + lo(k)}, CSlice{gather_[p][m][k].get(), 0}, len(k));
                }
    }

    // Every member's submission ended and its ticket taken on the recording thread, its queue call handed to the group's queue thread; the exchange's count, which every member's timeline reaches once its part of the exchange has run.
    // Each submission signals its member's timeline at the count and the peers' next submissions wait at it; with `beside` its copy queue's copies signal instead, waited at the lane's finish.
    uint64_t exchange(size_t lane, bool beside) {
        const size_t W = members_.size();
        std::vector<VulkanBackend::Submission> work(W);
        if (!submitter_) {
            // One member: its own queue call, nothing to exchange.
            work[0] = members_[0]->prepare({});
            taken_[0] = members_[0]->queued(work[0], work[0].queue(*members_[0]->device()));
            return 0;
        }
        // The group's open tagged batches go first, so no sender reaches its queue before its receivers.
        if (!submitter_->open_batches.empty()) close_open(*submitter_, nullptr, true);
        const uint64_t count = ++count_;
        std::vector<VulkanBackend::Submission> copies(beside ? W : 0);
        try {
            for (size_t m = 0; m < W; ++m) {
                std::vector<VkSemaphore> signals;
                if (!beside && signal_[m]) signals.push_back(signal_[m]);
                work[m] = members_[m]->prepare(signals, beside ? 0 : count);
                taken_[m] = members_[m]->queued(work[m], VK_SUCCESS);
                if (beside) {
                    copies[m] = members_[m]->copy_prepare(taken_[m], signal_[m], count);
                    copied_[lane][m] = copies[m].ticket;
                }
            }
        } catch (...) {
            // A ticket prepared and never handed to the queue thread is given up, so no wait for it hangs.
            for (size_t m = 0; m < W; ++m) {
                if (work[m].ticket) members_[m]->made(work[m].ticket, false);
                if (beside && copies[m].ticket) members_[m]->copy_made(copies[m], false);
            }
            throw;
        }
        if (!beside) wait_peers(count);
        const auto called = std::make_shared<bool>(false);
        submitter_->run({[this, called, work, copies] { queue_calls(submitter_->helpers(members_.size()), members_, work, *called, copies); },
                         [this, called, work, copies] {
                             if (*called) return;
                             for (size_t m = 0; m < work.size(); ++m) members_[m]->made(work[m].ticket, false);
                             for (size_t m = 0; m < copies.size(); ++m) members_[m]->copy_made(copies[m], false);
                         }});
        return count;
    }

    // Each member's next submission waits for its peers' timelines to reach `count`.
    void wait_peers(uint64_t count) {
        if (!submitter_) return;
        const size_t W = members_.size();
        for (size_t m = 0; m < W; ++m)
            for (size_t t = 0; t < W; ++t)
                if (t != m) members_[t]->wait_on(wait_[t][m], count);
    }

    // On the queue thread: the backends' queue calls, concurrently from three on, each settling its own ticket, all joined before return or throw.
    // `called` is set once nothing can throw before every ticket is settled, so the job's settle leaves them alone; a member's copies, where given, are queued once its work was and settled with it.
    static void queue_calls(core::JobThreads& threads, const std::vector<VulkanBackend*>& backs, const std::vector<VulkanBackend::Submission>& work, bool& called, const std::vector<VulkanBackend::Submission>& copies = {}) {
        const size_t W = backs.size();
        std::vector<VkResult> result(W, VK_ERROR_OUT_OF_HOST_MEMORY);
        const std::function<void(size_t)> call = [&](size_t m) {
            try {
                result[m] = queue_probe_ ? queue_probe_(*backs[m], false) : VK_SUCCESS;
                if (result[m] == VK_SUCCESS) result[m] = work[m].queue(*backs[m]->device());
                else if (work[m].release) work[m].release->store(work[m].release_seq);
            } catch (...) {
                result[m] = VK_ERROR_OUT_OF_HOST_MEMORY;
            }
            backs[m]->made(work[m].ticket, result[m] == VK_SUCCESS);
            if (copies.empty()) return;
            if (result[m] == VK_SUCCESS) {
                try {
                    result[m] = copies[m].queue(*backs[m]->device());
                } catch (...) {
                    result[m] = VK_ERROR_OUT_OF_HOST_MEMORY;
                }
            }
            backs[m]->copy_made(copies[m], result[m] == VK_SUCCESS);
        };
        called = true;
        if (W < 3) {
            for (size_t m = 0; m < W; ++m) call(m);
        } else {
            threads.run(W, call);
        }
        for (size_t m = 0; m < W; ++m) check(result[m], "vkQueueSubmit");
    }

    // A timeline a member signals once a submission, exported once as an opaque descriptor to each peer, which imports it for good as the semaphore its submissions wait on.
    void make_semaphores() {
        const size_t W = members_.size();
        count_ = 0;
        signal_.assign(W, VK_NULL_HANDLE);
        wait_.assign(W, std::vector<VkSemaphore>(W, VK_NULL_HANDLE));
        for (size_t m = 0; m < W; ++m) {
            if (W > 1) signal_[m] = semaphore(*members_[m]->device());
            for (size_t t = 0; t < W; ++t) {
                if (t == m) continue;
                wait_[t][m] = semaphore(*members_[t]->device());
                const Device& from = *members_[m]->device();
                VkSemaphoreGetFdInfoKHR gi{};
                gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
                gi.semaphore = signal_[m];
                gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
                int fd = -1;
                check(from.get_semaphore_fd(from.device, &gi, &fd), "vkGetSemaphoreFdKHR");
                const Device& to = *members_[t]->device();
                VkImportSemaphoreFdInfoKHR ii{};
                ii.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                ii.semaphore = wait_[t][m];
                ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
                ii.fd = fd;
                const VkResult r = to.import_semaphore_fd(to.device, &ii);
#if !defined(_WIN32)
                if (r != VK_SUCCESS && fd >= 0) close(fd);
#endif
                check(r, "vkImportSemaphoreFdKHR");
            }
        }
    }
    static VkSemaphore semaphore(const Device& d) {
        VkSemaphoreTypeCreateInfo ti{};
        ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkExportSemaphoreCreateInfo ex{};
        ex.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        ex.pNext = &ti;
        ex.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        si.pNext = &ex;
        VkSemaphore s = VK_NULL_HANDLE;
        check(d.fn.vkCreateSemaphore(d.device, &si, nullptr, &s), "vkCreateSemaphore");
        return s;
    }
    // On the recording thread given a failed queue call: each member's given-up tickets settled (`give_up`) and the semaphores made again.
    // The group's open batches are dropped first, so a member's flush in give_up prepares no member whose waits are not yet cleared.
    void recover() noexcept {
        for (VulkanCollective* o : submitter_->open_batches) o->drop_batch();
        submitter_->open_batches.clear();
        for (VulkanBackend* m : members_) {
            m->give_up();
            m->copies_abandon();
        }
        sent_[0] = sent_[1] = -1;
        destroy_semaphores();
        try { make_semaphores(); } catch (const std::exception&) { destroy_semaphores(); }
    }
    // The queue thread's calls go first, then the member threads: release drains members they call; the context that holds a collective goes before the model's backends.
    // Safe on what a throwing constructor left, each step undoing only what was made and each member routed given back; a pending queue-thread failure is kept (docs/src/backends-vulkan.md).
    void dismantle() noexcept {
        if (!runs_.empty()) {
            try { close_open(*submitter_, nullptr, false); } catch (...) {}
        }
        if (submitter_) {
            submitter_->disown(this);
            submitter_->drain();
            if (const std::exception_ptr e = submitter_->pending()) {
                for (VulkanBackend* m : members_) m->give_up();
                if (!submitter_->owned()) {
                    try { std::rethrow_exception(e); } catch (const std::exception& x) {
                        std::fprintf(stderr, "vulkan: %s; the tensor group ended before it was given out\n", x.what());
                    } catch (...) {}
                }
            }
        }
        release();
        for (size_t t = 0; t < pools_.size(); ++t) {
            const Device& d = *members_[t]->device();
            if (pools_[t]) d.fn.vkDestroyCommandPool(d.device, pools_[t], nullptr);
            if (t < rtimeline_.size()) d.fn.vkDestroySemaphore(d.device, rtimeline_[t], nullptr);
        }
        pools_.clear();
        rcmds_.clear();
        rtimeline_.clear();
        for (size_t m = 0; m < routed_; ++m) members_[m]->submit_alone();
        routed_ = 0;
    }
    // Every member drained first, so no submission still waits on, signals or reads what goes.
    // An open tagged batch closes with the group's others, whose pushes share its members' streams; one whose receivers cannot be recorded is dropped, and its senders abort at the cap.
    void release() noexcept {
        if (!runs_.empty()) {
            try { close_open(*submitter_, nullptr, false); } catch (...) {}
        }
        for (VulkanBackend* m : members_) m->sync();
        if (submitter_) submitter_->drain();
        for (size_t t = 0; t < rtimeline_.size(); ++t) wait_received(t, received_);
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
    int lane_parity_[2] = {0, 0};
    int sent_[2] = {-1, -1};                  // per lane, the inbox parity its sent sum's copies went to, -1 once finished
    uint64_t sent_count_[2] = {0, 0};         // per lane, the exchange count of its sent sum, which the peers wait for at its finish
    std::vector<uint64_t> copied_[2];         // per lane and member, the copy submission of its last sent sum, which its finish waits for alone
    uint64_t count_ = 0;   // exchanges made since the semaphores were; every member's timeline reaches it once its submission of that exchange finishes
    std::vector<Ticket> taken_, last_, before_;   // per member, its submission of the last exchange, of the last sum and of the sum before it
    std::vector<VkBufferMemoryBarrier> acquires_;   // recording-thread scratch, sized once for every peer
    static inline VkResult (*queue_probe_)(const VulkanBackend&, bool receivers) = nullptr;   // tests only: called before each queue call, a failure it returns standing in for the call's
    static inline std::atomic<size_t> host_signals_{0};   // receivers given up by a host signal, which tests count
    std::vector<BufferPtr> partial_, scratch_;
    std::vector<std::vector<std::shared_ptr<VulkanBuffer>>> inbox_[kInboxes];   // [parity][owner][sender]: the owner's inbox for that sender's partial rows
    std::vector<std::vector<BufferPtr>> imported_[kInboxes];                    // [parity][owner][sender]: that inbox on the sender, its only importer
    size_t share_align_ = 1;   // the members' largest storage-buffer offset alignment, in floats
    std::vector<std::vector<std::shared_ptr<VulkanBuffer>>> gather_[2];  // [parity][owner][sender]: the owner's gather rows for that sender's share, where a sum may take two shots
    std::vector<std::vector<BufferPtr>> gathered_[2];                    // [parity][owner][sender]: those gather rows on the sender, their only importer
    std::vector<VkSemaphore> signal_;                  // one timeline per source, imported by every peer
    std::vector<std::vector<VkSemaphore>> wait_;        // [member][peer]: that peer's timeline as member imported it
    bool beside_ = false;                               // every member has a copy queue, so a lane's copies run beside the next part's work
    std::shared_ptr<VulkanSubmitter> submitter_;        // the group's queue thread, from two members on
    size_t routed_ = 0;                                 // members routed through submitter_, given back on close

    // The tagged route (docs/STATUS.md, Vulkan collective route), empty where tag() found a member without what it needs.
    struct Run {
        size_t n;
        uint32_t base, epochs;   // the run's sums are epochs base + 1 to base + epochs
    };
    std::atomic<size_t> tagged_limit_{0};   // the largest sum in place it takes, zero while it is off; a give-up on the queue thread clears it
    uint32_t slots_ = 0, cap_ = 0;   // 16-byte slots a tagged row; the spin bound in device clock ticks, a second
    uint32_t epoch_ = 8, gen_ = 1;   // the last epoch recorded, which only grows; the open batch's generation
    uint32_t reported_ = 0;          // the latest aborted generation a wait read
    std::atomic<bool> aborted_{false};
    size_t tagged_sums_ = 0;
    std::vector<Run> runs_;          // the open batch's receiver runs, empty with no batch open
    uint64_t received_ = 0;          // batches closed, the value each member's receive timeline reaches once that batch's receivers finish
    uint64_t rqueued_[kReceivers] = {};
    std::vector<uint64_t> rsent_;    // per member, the latest batch whose receivers reached its receive queue; written on the queue thread, read by a settle there or after it
    std::vector<std::shared_ptr<VulkanBuffer>> ctl_, report_;   // per member: its control words (acks, own epochs, abort), uncached; their first words where the host reads them
    std::vector<std::vector<std::shared_ptr<VulkanBuffer>>> tags_;   // [owner][sender]: the owner's tagged rows for that sender
    std::vector<std::vector<BufferPtr>> tagged_in_;                  // [owner][sender]: those rows on the sender, their only importer
    std::vector<VkCommandPool> pools_;   // per member, on its receive family
    std::vector<std::vector<VkCommandBuffer>> rcmds_;
    std::vector<VkSemaphore> rtimeline_;
};

inline Ticket VulkanBackend::close_batches(bool report) { return VulkanCollective::close_open(*submitter_, this, report); }

// A collective over Vulkan devices of one profile, each opened once, that share memory as dma-buf and semaphores as sync files, under any PCI root complexes; none where a device lacks them, as on Windows.
std::unique_ptr<Collective> VulkanBackend::join(const std::vector<Backend*>& members, size_t rows, size_t width) {
    if (members.empty() || members[0] != this) throw std::logic_error("backend: a collective is joined by its first member");
    if (members.size() > kMaxTensorWidth) throw std::runtime_error("backend: a Vulkan collective of more than " + std::to_string(kMaxTensorWidth) + " devices");
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
        return std::make_unique<VulkanCollective>(group, rows, width, VulkanBackend::tagged_crossover(group.size()));
    } catch (const std::exception& e) {
        std::string names;
        for (const VulkanBackend* v : group) names += (names.empty() ? "" : ", ") + v->dev_->caps.device + (v->dev_->pci_root.empty() ? "" : " under " + v->dev_->pci_root);
        throw std::runtime_error("backend: a tensor group of " + names + " cannot share its memory and semaphores: " + e.what());
    }
}

} // namespace
} // namespace backend
