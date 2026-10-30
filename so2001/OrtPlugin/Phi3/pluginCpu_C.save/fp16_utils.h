// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <string.h>
#include <math.h>

static inline float FusionCpuHalfToFloat(uint16_t h) {
    const uint16_t h_exp = (uint16_t)((h & 0x7C00u) >> 10);
    const uint16_t h_mant = (uint16_t)(h & 0x03FFu);
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_mant == 0) {
            f = sign;
        } else {
            const float value = ldexpf((float)h_mant / 1024.0f, -14);
            return sign ? -value : value;
        }
    } else if (h_exp == 31) {
        f = sign | 0x7F800000u | ((uint32_t)h_mant << 13);
    } else {
        const uint32_t exp = (uint32_t)h_exp + (127u - 15u);
        f = sign | (exp << 23) | ((uint32_t)h_mant << 13);
    }
    {
        float value;
        memcpy(&value, &f, sizeof(value));
        return value;
    }
}

static inline uint16_t FusionCpuFloatToHalf(float value) {
    union {
        uint32_t u;
        float f;
    } v;
    uint32_t sign;
    int32_t exp;
    uint32_t mant;
    uint16_t hsign;

    v.f = value;
    sign = (v.u >> 31) & 1u;
    exp = (int32_t)((v.u >> 23) & 0xFFu) - 127;
    mant = v.u & 0x7FFFFFu;
    hsign = (uint16_t)(sign << 15);
    if (((v.u >> 23) & 0xFFu) == 0xFFu) {
        return (uint16_t)(hsign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp > 15) return (uint16_t)(hsign | 0x7C00u);
    if (exp > -15) {
        uint16_t hexp;
        uint16_t hmant;
        const uint32_t round_bit = (mant >> 12) & 1u;
        exp += 15;
        hexp = (uint16_t)(exp & 0x1F);
        hmant = (uint16_t)(mant >> 13);
        if (round_bit && ((hmant & 1u) || (mant & 0xFFFu))) {
            ++hmant;
            if (hmant == 0x400) {
                hmant = 0;
                ++hexp;
                if (hexp == 0x1F) return (uint16_t)(hsign | 0x7C00u);
            }
        }
        return (uint16_t)(hsign | (hexp << 10) | (hmant & 0x3FF));
    }
    if (exp >= -24) {
        const int shift = (-exp) - 14;
        uint32_t mant_norm = (mant | 0x800000u) >> (shift + 13);
        const uint32_t remain = (mant | 0x800000u) & ((1u << (shift + 13)) - 1u);
        if (remain > (1u << (shift + 12)) ||
            (remain == (1u << (shift + 12)) && (mant_norm & 1u))) {
            ++mant_norm;
        }
        return (uint16_t)(hsign | (mant_norm & 0x3FF));
    }
    return hsign;
}
