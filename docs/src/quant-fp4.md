# `src/quant/fp4.hpp` - shared FP4 codebook

`quant::FP4_VALUES` owns the doubled E2M1 signed-integer codebook used by MXFP4 and NVFP4. Both zero codes map to integer zero. Each format owns its scale interpretation, block layout and output arithmetic in its decoder; sharing the table does not unify those layouts or establish native FP4 hardware support.
