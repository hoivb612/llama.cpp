// One wave reuses a weight row across a small batch of F32 activations.
#include "ggml_common.hlsli"

#ifdef WAVE_SIZE
#define GROUP_SIZE WAVE_SIZE
#else
#define GROUP_SIZE 64
#endif
#define MAX_M 12
#ifndef SMALL_F16
#define SMALL_F16 0
#endif

uint read_q8_word(uint off) {
    const uint shift = (off & 3u) * 8u;
    const uint lo = src0.Load(off & ~3u);
    const uint hi = src0.Load((off & ~3u) + (shift == 0u ? 0u : 4u));
    return shift == 0u ? lo : (lo >> shift) | (hi << (32u - shift));
}

float4 unpack_q8(uint q) {
    return float4(int4((int)(q << 24) >> 24, (int)(q << 16) >> 24,
                      (int)(q << 8) >> 24, (int)q >> 24));
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint row = group_x_2d(gid);
    if (row >= ne0) {
        return;
    }
    const uint base = src0_offset + row * nb01;
    precise float acc[MAX_M];
    [unroll] for (uint m = 0; m < MAX_M; ++m) {
        acc[m] = 0.0f;
    }
    for (uint k = tid * 8u; k < ne00; k += GROUP_SIZE * 8u) {
#if SMALL_F16
        const uint4 w = src0.Load4(base + k * 2u);
        const float d = 1.0f;
        const float4 q0 = float4(f16_to_f32(w.x & 0xffffu), f16_to_f32(w.x >> 16),
                                f16_to_f32(w.y & 0xffffu), f16_to_f32(w.y >> 16));
        const float4 q1 = float4(f16_to_f32(w.z & 0xffffu), f16_to_f32(w.z >> 16),
                                f16_to_f32(w.w & 0xffffu), f16_to_f32(w.w >> 16));
#else
        const uint block = base + (k / 32u) * 34u;
        const uint word = src0.Load(block & ~3u);
        const float d = f16_to_f32((word >> ((block & 2u) * 8u)) & 0xffffu);
        const uint off = block + 2u + (k & 31u);
        const float4 q0 = unpack_q8(read_q8_word(off));
        const float4 q1 = unpack_q8(read_q8_word(off + 4u));
#endif
        [unroll] for (uint m = 0; m < MAX_M; ++m) {
            if (m < ne1) {
                const uint x = src1_offset + m * nb11 + k * 4u;
                const float4 x0 = asfloat(src1.Load4(x));
                const float4 x1 = asfloat(src1.Load4(x + 16u));
                acc[m] = mad(d, dot(q0, x0) + dot(q1, x1), acc[m]);
            }
        }
    }
    [unroll] for (uint m = 0; m < MAX_M; ++m) {
        if (m < ne1) {
            const float sum = WaveActiveSum(acc[m]);
            if (WaveIsFirstLane()) {
                dst.Store(dst_offset + row * nb0 + m * nb1, asuint(sum));
            }
        }
    }
}
