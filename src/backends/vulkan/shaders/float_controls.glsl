// Only modules selected on devices reporting both 32-bit preservation properties use these modes.
#ifdef LLMX_PRESERVE_F32
#extension GL_EXT_spirv_intrinsics : require
// DenormPreserve and SignedZeroInfNanPreserve, respectively.
spirv_execution_mode(capabilities = [4464], 4459, 32);
spirv_execution_mode(capabilities = [4466], 4461, 32);
#endif
