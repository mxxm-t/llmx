#include "backends/vulkan/vulkan_backend.cpp"
#include <iostream>
#include <new>
namespace { thread_local bool fail_new = false; }
void* operator new(std::size_t n) {
    if (fail_new) { fail_new = false; throw std::bad_alloc(); }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
namespace backend { namespace {
struct VulkanLifetimeTest {
    static std::shared_ptr<Device> device(VulkanBackend& b) { return b.dev_; }
    static void kernel(VulkanBackend& b) { b.kernel(K_ADD); }
    static void query(VulkanBackend& b, VkQueryPool q) { b.queries_ = q; }
    static void drop(VulkanBackend& b, VulkanBuffer& x) {
        b.open();
        std::vector<std::shared_ptr<VulkanBuffer>> pending;
        pending.reserve(1);
        b.pending_[b.ring_index_].swap(pending);
        fail_new = true;
        b.drop_padded(x);
    }
};
}}
namespace {
int modules=0, sets=0, layouts=0, pipes=0, query_destroys=0;
uintptr_t next_handle=1024;
bool pipeline_failure=true;
template<class T> T handle() { return (T)++next_handle; }
VKAPI_ATTR VkResult VKAPI_CALL shader(VkDevice,const VkShaderModuleCreateInfo*,const VkAllocationCallbacks*,VkShaderModule* x) { ++modules; *x=handle<VkShaderModule>(); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL set(VkDevice,const VkDescriptorSetLayoutCreateInfo*,const VkAllocationCallbacks*,VkDescriptorSetLayout* x) { ++sets; *x=handle<VkDescriptorSetLayout>(); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL layout(VkDevice,const VkPipelineLayoutCreateInfo*,const VkAllocationCallbacks*,VkPipelineLayout* x) { ++layouts; *x=handle<VkPipelineLayout>(); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL pipeline(VkDevice,VkPipelineCache,uint32_t,const VkComputePipelineCreateInfo*,const VkAllocationCallbacks*,VkPipeline* x) { if (pipeline_failure) { *x=VK_NULL_HANDLE; return VK_ERROR_OUT_OF_HOST_MEMORY; } ++pipes; *x=handle<VkPipeline>(); return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL shader_drop(VkDevice,VkShaderModule,const VkAllocationCallbacks*) { --modules; }
VKAPI_ATTR void VKAPI_CALL set_drop(VkDevice,VkDescriptorSetLayout,const VkAllocationCallbacks*) { --sets; }
VKAPI_ATTR void VKAPI_CALL layout_drop(VkDevice,VkPipelineLayout,const VkAllocationCallbacks*) { --layouts; }
VKAPI_ATTR void VKAPI_CALL pipeline_drop(VkDevice,VkPipeline,const VkAllocationCallbacks*) { --pipes; }
VKAPI_ATTR void VKAPI_CALL query_drop(VkDevice,VkQueryPool,const VkAllocationCallbacks*) { ++query_destroys; }
}
int main() {
    using namespace backend;
    int reproduced=0;
    {
        auto base=make_vulkan_backend(0);
        auto& b=dynamic_cast<VulkanBackend&>(*base);
        auto dev=VulkanLifetimeTest::device(b);
        auto original=dev->fn;
        dev->fn.vkCreateShaderModule=shader; dev->fn.vkDestroyShaderModule=shader_drop;
        dev->fn.vkCreateDescriptorSetLayout=set; dev->fn.vkDestroyDescriptorSetLayout=set_drop;
        dev->fn.vkCreatePipelineLayout=layout; dev->fn.vkDestroyPipelineLayout=layout_drop;
        dev->fn.vkCreateComputePipelines=pipeline; dev->fn.vkDestroyPipeline=pipeline_drop;
        bool threw=false;
        try { VulkanLifetimeTest::kernel(b); } catch(const std::runtime_error&) { threw=true; }
        pipeline_failure=false;
        VulkanLifetimeTest::kernel(b);
        base.reset();
        std::cout << "pipeline_retry threw=" << threw << " unreleased_modules=" << modules << " unreleased_sets=" << sets << " unreleased_layouts=" << layouts << " unreleased_pipelines=" << pipes << '\n';
        reproduced += threw && modules==1 && sets==1 && layouts==1 && pipes==0;
        dev->fn=original;
    }
    {
        auto base=make_vulkan_backend(0);
        auto& b=dynamic_cast<VulkanBackend&>(*base);
        auto dev=VulkanLifetimeTest::device(b);
        auto original=dev->fn;
        dev->fn.vkDestroyQueryPool=query_drop;
        VulkanLifetimeTest::query(b,handle<VkQueryPool>());
        base.reset();
        std::cout << "diagnostic_query destructor_calls=" << query_destroys << '\n';
        reproduced += query_destroys==0;
        dev->fn=original;
    }
    {
        auto base=make_vulkan_backend(0);
        auto& b=dynamic_cast<VulkanBackend&>(*base);
        auto weights=b.alloc(2048,Memory::host_visible);
        auto& w=as_vulkan(*weights);
        w.adopted=true;
        auto dev=VulkanLifetimeTest::device(b);
        w.padded.emplace(0,VulkanBuffer::Padded{1,256,std::make_shared<VulkanBuffer>(dev,1040,true)});
        w.padded.emplace(1024,VulkanBuffer::Padded{1,256,std::make_shared<VulkanBuffer>(dev,1040,true)});
        bool threw=false;
        try { VulkanLifetimeTest::drop(b,w); } catch(const std::bad_alloc&) { threw=true; }
        fail_new=false;
        size_t nulls=0;
        for (const auto& e:w.padded) if (!e.second.copy) ++nulls;
        std::cout << "drop_padded threw=" << threw << " retained_entries=" << w.padded.size() << " null_cached_copies=" << nulls << '\n';
        reproduced += threw && nulls==1 && w.padded.size()==2;
        b.sync();
    }
    std::cout << "review_findings_reproduced=" << reproduced << "/3\n";
    return reproduced==3 ? 0 : 1;
}
