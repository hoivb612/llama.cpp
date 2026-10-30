// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// fp16_utils.h
//
// fp16 <-> fp32 conversion helpers (manual bit-twiddling, NOT hardware
// intrinsics) that match the run_phi3 harness numerics exactly. Extracted to a
// shared header so the baseline kernel TU (fusion_cpu_ep.cpp) and the
// /arch:AVX512 kernel TU (matmulnbits_avx512.cpp) round identically -- the
// AVX-512 path must produce numerically-equivalent results to the scalar path,
// so both must use the SAME scale-conversion code.

#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace fusion_cpu_ep {

inline float HalfToFloat(uint16_t h) {
    const uint16_t h_exp  = (h & 0x7C00u) >> 10;
    const uint16_t h_mant =  h & 0x03FFu;
    const uint32_t sign   = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0) {  // subnormal / zero
        if (h_mant == 0) {
            f = sign;
        } else {
            const float val = std::ldexp(static_cast<float>(h_mant) / 1024.0f, -14);
            return sign ? -val : val;
        }
    } else if (h_exp == 31) {  // inf / NaN
        f = sign | 0x7F800000u | (static_cast<uint32_t>(h_mant) << 13);
    } else {  // normal
        const uint32_t exp = static_cast<uint32_t>(h_exp) + (127u - 15u);
        f = sign | (exp << 23) | (static_cast<uint32_t>(h_mant) << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline uint16_t FloatToHalf(float value) {
    union { uint32_t u; float f; } v;
    v.f = value;
    const uint32_t x = v.u;
    const uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127;
    const uint32_t mant = x & 0x7FFFFFu;
    const uint16_t hsign = static_cast<uint16_t>(sign << 15);
    if (((x >> 23) & 0xFFu) == 0xFFu) {  // Inf / NaN
        return static_cast<uint16_t>(hsign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp > 15) {  // overflow -> Inf
        return static_cast<uint16_t>(hsign | 0x7C00u);
    } else if (exp > -15) {  // normalized half
        exp += 15;
        uint16_t hexp  = static_cast<uint16_t>(exp & 0x1F);
        uint16_t hmant = static_cast<uint16_t>(mant >> 13);
        const uint32_t round_bit = (mant >> 12) & 1u;
        if (round_bit && ((hmant & 1u) || (mant & 0xFFFu))) {
            ++hmant;
            if (hmant == 0x400) {
                hmant = 0;
                ++hexp;
                if (hexp == 0x1F) return static_cast<uint16_t>(hsign | 0x7C00u);
            }
        }
        return static_cast<uint16_t>(hsign | (hexp << 10) | (hmant & 0x3FF));
    } else if (exp >= -24) {  // subnormal half
        const int shift = (-exp) - 14;
        uint32_t mant_norm = (mant | 0x800000u) >> (shift + 13);
        const uint32_t remain = (mant | 0x800000u) & ((1u << (shift + 13)) - 1u);
        if (remain > (1u << (shift + 12)) ||
            (remain == (1u << (shift + 12)) && (mant_norm & 1u))) {
            ++mant_norm;
        }
        return static_cast<uint16_t>(hsign | (mant_norm & 0x3FF));
    }
    return hsign;  // underflow -> zero
}

}  // namespace fusion_cpu_ep
