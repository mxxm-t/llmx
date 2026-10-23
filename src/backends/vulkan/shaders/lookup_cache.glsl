// Every invocation loads its dictionary words before any early return in the caller.
shared uint lookup_words[LOOKUP_GRID_SIZE];
void lookup_init() {
    for (uint i = gl_LocalInvocationIndex; i < LOOKUP_GRID_SIZE; i += gl_WorkGroupSize.x)
        lookup_words[i] = LOOKUP_GRID_SOURCE(i);
    barrier();
}
