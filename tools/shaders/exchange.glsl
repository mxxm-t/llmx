// The exchange of llmx-vk-handoff (tools/vulkan_handoff.cpp): every member writes its partial into slot `member` of each member's inbox, and each member adds the slots in member order.
// An inbox holds, per parity of the epoch, a slot of n floats for each member, then a flag a member for each parity, 64 words apart.
// With LLMX_MM the stores and loads are Vulkan memory model atomics at scope LLMX_SCOPE, which the flag wait needs; without it they are plain, and a submission boundary orders them.

layout(push_constant) uniform Args { uint n; uint epoch; uint member; uint width; } pc;

// A member's partial in an epoch, chosen so every sum is exact in F32 and differs between epochs.
float partial(uint member, uint e, uint i) { return float(member + 1u) * float(e & 1023u) + float(i % 97u) * 0.5; }

uint slot(uint member, uint i) { return ((pc.epoch & 1u) * pc.width + member) * pc.n + i; }
uint flag(uint member) { return 2u * pc.width * pc.n + ((pc.epoch & 1u) * pc.width + member) * 64u; }
