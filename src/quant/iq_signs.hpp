#pragma once
#include <cstdint>

namespace quant {

// Store seven sign bits and set the eighth so the full byte has even parity.
inline uint8_t iq_even_parity_signs(unsigned code) {
    code &= 127u;
    unsigned parity = code;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    return uint8_t(code | ((parity & 1u) << 7));
}

} // namespace quant
