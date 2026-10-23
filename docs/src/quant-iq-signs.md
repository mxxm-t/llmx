# `src/quant/iq_signs.hpp` - shared IQ sign expansion

`iq_even_parity_signs` expands the seven explicit sign bits used by IQ2_XXS, IQ2_XS and IQ3_XXS. It sets bit 7 when the low seven bits have odd population count, making the full sign byte even parity. The helper centralizes this format rule without storing a sign lookup table.

The IQ decoders apply each resulting bit as a sign after scaling the selected dictionary magnitude. Tests sweep all 128 codes against independent field expectations. The helper does not define any quantized block layout or backend matrix path.
