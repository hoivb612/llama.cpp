// Shared tiled GEMM based on Vulkan mul_mm.comp and mul_mm_funcs.glsl.
// Compose native tiles into 32x32 wave matrices.
#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#if !defined(IW_TILED_F16) && !defined(IW_TILED_BF16)
#include "quant_dequant.hlsli"
#endif

#ifndef VP_FULL_F16
#define VP_FULL_F16 0
#endif
#ifndef VP_NATIVE_F32
#define VP_NATIVE_F32 0
#endif
#ifndef VP_LDS_OUTPUT
#define VP_LDS_OUTPUT 0
#endif
#ifndef VP_B_F16
#define VP_B_F16 0
#endif
#define VP_TK 16
#define VP_BK 32

#ifndef VP_BM
#define VP_BM 128
#endif
#ifndef VP_BN
#define VP_BN 128
#endif
#define VP_WM 32
#define VP_WN 32
#define VP_PITCH (VP_BK + 8)
#define VP_THREADS ((VP_BM / VP_WM) * (VP_BN / VP_WN) * WAVE_SIZE)
#define VP_ELEMS (VP_WM * VP_WN / WAVE_SIZE)

typedef Matrix<ComponentType::F16, VP_WM, VP_TK, MatrixUse::A, MatrixScope::Wave> VpA;
typedef Matrix<ComponentType::F16, VP_TK, VP_WN, MatrixUse::B, MatrixScope::Wave> VpB;
#if VP_NATIVE_F32
typedef Matrix<ComponentType::F32, VP_WM, VP_WN, MatrixUse::Accumulator, MatrixScope::Wave> VpC;
#else
typedef Matrix<ComponentType::F16, VP_WM, VP_WN, MatrixUse::Accumulator, MatrixScope::Wave> VpC;
#endif
typedef Matrix<ComponentType::F32, VP_WM, VP_WN, MatrixUse::Accumulator, MatrixScope::Wave> VpOut;

groupshared float16_t vp_a[VP_BM * VP_PITCH];
groupshared float16_t vp_b[VP_BN * VP_PITCH];
#if VP_LDS_OUTPUT
groupshared float vp_c[VP_WM * VP_WN];
#endif

#if defined(MMID_Q8_0) || defined(MMID_Q5_0) || defined(MMID_Q6_K) || defined(MMID_Q4_0) || defined(MMID_Q4_1) || defined(MMID_Q5_1) || defined(MMID_IQ4_NL)
uint tiled_read_u32(uint address) {
    const uint lo = src0.Load(address & ~3u);
#if VP_NATIVE_F32
    const uint hi = src0.Load((address & ~3u) + ((address & 2u) == 0 ? 0u : 4u));
    return (address & 2u) == 0 ? lo : (lo >> 16) | (hi << 16);
#else
    if ((address & 2u) == 0) {
        return lo;
    }
    return (lo >> 16) | (src0.Load((address & ~3u) + 4) << 16);
#endif
}
#endif

#if defined(MMID_MXFP4)
uint tiled_read_u32_bytes(uint address) {
    const uint lo = src0.Load(address & ~3u);
    const uint shift = (address & 3u) * 8u;
    if (shift == 0) {
        return lo;
    }
    // Use the last requested byte so an aligned tail cannot read the next DWORD.
    return (lo >> shift) | (src0.Load((address + 3) & ~3u) << (32 - shift));
}
#endif

WAVE_SIZE_ATTR
[numthreads(VP_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint wave = GetGroupWaveIndex();
    const uint warp_r = wave % (VP_BM / VP_WM);
    const uint warp_c = wave / (VP_BM / VP_WM);
    const uint row = group.x * VP_BM;
    const uint col = group.y * VP_BN;
    const uint i2 = group.z % ne2;
    const uint i3 = group.z / ne2;
    const uint a_base = src0_offset + (i2 * ne02 / ne2) * nb02 + (i3 * ne03 / ne3) * nb03;
    const uint b_base = src1_offset + i2 * nb12 + i3 * nb13;
    const uint d_base = dst_offset + i2 * nb2 + i3 * nb3;

#if VP_NATIVE_F32
    VpC sum = VpC::Splat(0.0f);
#else
    VpC sum = VpC::Splat((float16_t)0);
#endif
#if !VP_FULL_F16 && !VP_NATIVE_F32
    float totals[VP_ELEMS];
    [unroll] for (uint e = 0; e < VP_ELEMS; ++e) {
        totals[e] = 0.0f;
    }
#endif

    for (uint k0 = 0; k0 < ne00; k0 += VP_BK) {
#if defined(MMID_Q8_0)
        [unroll] for (uint r = 0; r < VP_BM; r += VP_THREADS * 8 / VP_BK) {
            const uint m = tid / (VP_BK / 8) + r;
            const uint k = (tid % (VP_BK / 8)) * 8;
            const uint base = a_base + (row + m) * nb01 + (k0 / 32) * MMID_BLOCK_SIZE;
            const float d = mmid_read_f16(src0, base);
            [unroll] for (uint j = 0; j < 2; ++j) {
                const uint q = tiled_read_u32(base + 2 + k + j * 4);
                const uint o = m * VP_PITCH + k + j * 4;
                vp_a[o + 0] = (float16_t)(d * (float)(int(q << 24) >> 24));
                vp_a[o + 1] = (float16_t)(d * (float)(int(q << 16) >> 24));
                vp_a[o + 2] = (float16_t)(d * (float)(int(q << 8) >> 24));
                vp_a[o + 3] = (float16_t)(d * (float)(int(q) >> 24));
            }
        }
#else
        [unroll] for (uint r = 0; r < VP_BM; r += VP_THREADS * 4 / VP_BK) {
            const uint m = tid / (VP_BK / 4) + r;
            const uint k = (tid % (VP_BK / 4)) * 4;
            const uint kk = k0 + k;
            const uint row_off = a_base + (row + m) * nb01;
            const uint dst_a = m * VP_PITCH + k;
#if defined(IW_TILED_F16)
            const uint2 packed = src0.Load2(row_off + kk * 2);
            vp_a[dst_a + 0] = asfloat16((uint16_t)packed.x);
            vp_a[dst_a + 1] = asfloat16((uint16_t)(packed.x >> 16));
            vp_a[dst_a + 2] = asfloat16((uint16_t)packed.y);
            vp_a[dst_a + 3] = asfloat16((uint16_t)(packed.y >> 16));
#elif defined(IW_TILED_BF16)
            const uint2 packed = src0.Load2(row_off + kk * 2);
            vp_a[dst_a + 0] = (float16_t)asfloat(packed.x << 16);
            vp_a[dst_a + 1] = (float16_t)asfloat(packed.x & 0xffff0000u);
            vp_a[dst_a + 2] = (float16_t)asfloat(packed.y << 16);
            vp_a[dst_a + 3] = (float16_t)asfloat(packed.y & 0xffff0000u);
#elif defined(MMID_Q4_K) || defined(MMID_Q5_K)
            const uint block = row_off + (kk / 256) * MMID_BLOCK_SIZE;
            const uint s = (kk % 256) / 32;
            const uint4 header = src0.Load4(block);
            const uint dm = header.x;
            const uint3 scales = header.yzw;
            const uint offset = (s & 3) * 8;
            const uint sc0 = s < 4 ? scales.x : scales.z;
            const uint mn0 = s < 4 ? scales.y : scales.z;
            const uint shift1 = s < 4 ? offset : offset + 2;
            const uint scale = ((sc0 >> offset) & 15) | ((scales.x >> shift1) & 48);
            const uint minimum = ((mn0 >> (s < 4 ? offset : offset + 4)) & 15) |
                                 ((scales.y >> shift1) & 48);
            const float d = f16_to_f32(dm & 65535) * (float)scale;
            const float mn = -f16_to_f32(dm >> 16) * (float)minimum;
#if defined(MMID_Q4_K)
            const uint q = (src0.Load(block + 16 + (s >> 1) * 32 + (kk % 32)) >> ((s & 1) * 4)) & 0x0f0f0f0f;
#else
            const uint lo = (src0.Load(block + 48 + (s >> 1) * 32 + (kk % 32)) >> ((s & 1) * 4)) & 0x0f0f0f0f;
            const uint hi = ((src0.Load(block + 16 + (kk % 32)) >> s) & 0x01010101) << 4;
            const uint q = lo | hi;
#endif
            vp_a[dst_a + 0] = (float16_t)mad(d, (float)(q & 255), mn);
            vp_a[dst_a + 1] = (float16_t)mad(d, (float)((q >> 8) & 255), mn);
            vp_a[dst_a + 2] = (float16_t)mad(d, (float)((q >> 16) & 255), mn);
            vp_a[dst_a + 3] = (float16_t)mad(d, (float)(q >> 24), mn);
#elif defined(MMID_Q5_0)
            const uint block = row_off + (kk / 32) * MMID_BLOCK_SIZE;
            const uint elem = kk % 32;
            const float d = mmid_read_f16(src0, block);
            const uint qh = tiled_read_u32(block + 2) >> elem;
            const uint ql = (tiled_read_u32(block + 6 + elem % 16) >> (elem < 16 ? 0 : 4)) & 0x0f0f0f0f;
            [unroll] for (uint e = 0; e < 4; ++e) {
                const int q = (int)(((ql >> (8 * e)) & 15) | (((qh >> e) & 1) << 4)) - 16;
                vp_a[dst_a + e] = (float16_t)(d * (float)q);
            }
#elif defined(MMID_Q6_K)
            const uint block = row_off + (kk / 256) * MMID_BLOCK_SIZE;
            const uint elem = kk % 256;
            const uint ip = elem / 128;
            const uint il = elem % 128;
            const float d = mmid_read_f16(src0, block + 208) *
                            (float)mmid_read_sbyte(src0, block + 192 + 8 * ip + il / 16);
            const uint ql = (tiled_read_u32(block + 64 * ip + il % 64) >> (il < 64 ? 0 : 4)) & 0x0f0f0f0f;
            const uint qh = (tiled_read_u32(block + 128 + 32 * ip + il % 32) >> (2 * (il / 32))) & 0x03030303;
            const uint q = ql | (qh << 4);
            [unroll] for (uint e = 0; e < 4; ++e) {
                vp_a[dst_a + e] = (float16_t)(d * (float)((int)((q >> (8 * e)) & 63) - 32));
            }
#elif defined(MMID_Q4_0) || defined(MMID_IQ4_NL)
            const uint block = row_off + (kk / 32) * MMID_BLOCK_SIZE;
            const uint elem = kk % 32;
            const float d = mmid_read_f16(src0, block);
            const uint qs = (tiled_read_u32(block + 2 + elem % 16) >> (elem < 16 ? 0 : 4)) & 0x0f0f0f0f;
            [unroll] for (uint e = 0; e < 4; ++e) {
                const uint q = (qs >> (8 * e)) & 15;
#if defined(MMID_Q4_0)
                vp_a[dst_a + e] = (float16_t)(d * (float)((int)q - 8));
#else
                vp_a[dst_a + e] = (float16_t)(d * (float)mmid_kvalues_iq4nl(q));
#endif
            }
#elif defined(MMID_Q4_1) || defined(MMID_Q5_1)
            const uint block = row_off + (kk / 32) * MMID_BLOCK_SIZE;
            const uint elem = kk % 32;
            const uint dm = tiled_read_u32(block);
            const float d = f16_to_f32(dm & 65535);
            const float minimum = f16_to_f32(dm >> 16);
#if defined(MMID_Q4_1)
            const uint qs = (tiled_read_u32(block + 4 + elem % 16) >> (elem < 16 ? 0 : 4)) & 0x0f0f0f0f;
#else
            const uint qh = tiled_read_u32(block + 4) >> elem;
            const uint qs = (tiled_read_u32(block + 8 + elem % 16) >> (elem < 16 ? 0 : 4)) & 0x0f0f0f0f;
#endif
            [unroll] for (uint e = 0; e < 4; ++e) {
#if defined(MMID_Q4_1)
                const uint q = (qs >> (8 * e)) & 15;
#else
                const uint q = ((qs >> (8 * e)) & 15) | (((qh >> e) & 1) << 4);
#endif
                vp_a[dst_a + e] = (float16_t)((float)q * d + minimum);
            }
#elif defined(MMID_MXFP4)
            const uint block = row_off + (kk / 32) * MMID_BLOCK_SIZE;
            const uint elem = kk % 32;
            const float d = mmid_e8m0_half(mmid_read_byte(src0, block));
            const uint qs = (tiled_read_u32_bytes(block + 1 + elem % 16) >> (elem < 16 ? 0 : 4)) & 0x0f0f0f0f;
            [unroll] for (uint e = 0; e < 4; ++e) {
                const uint q = (qs >> (8 * e)) & 15;
                vp_a[dst_a + e] = (float16_t)(d * (float)mmid_kvalues_fp4(q));
            }
#else
            [unroll] for (uint e = 0; e < 4; ++e) {
                vp_a[dst_a + e] = (float16_t)mmid_dequant(src0, row_off, kk + e);
            }
#endif
        }
#endif
        [unroll] for (uint r = 0; r < VP_BN; r += VP_THREADS * 8 / VP_BK) {
            const uint n = tid / (VP_BK / 8) + r;
            const uint k = (tid % (VP_BK / 8)) * 8;
#if VP_B_F16
            // src1 was converted to F16 by a pre-pass.
            const uint4 packed = src1.Load4(b_base + (col + n) * nb11 + (k0 + k) * 2);
            const uint dst_b = n * VP_PITCH + k;
            [unroll] for (uint j = 0; j < 4; ++j) {
                vp_b[dst_b + 2 * j + 0] = asfloat16((uint16_t)packed[j]);
                vp_b[dst_b + 2 * j + 1] = asfloat16((uint16_t)(packed[j] >> 16));
            }
#else
            const uint address = b_base + (col + n) * nb11 + (k0 + k) * 4;
            const float4 values = asfloat(src1.Load4(address));
            const float4 values_hi = asfloat(src1.Load4(address + 16));
            const uint dst_b = n * VP_PITCH + k;
            vp_b[dst_b + 0] = (float16_t)values.x;
            vp_b[dst_b + 1] = (float16_t)values.y;
            vp_b[dst_b + 2] = (float16_t)values.z;
            vp_b[dst_b + 3] = (float16_t)values.w;
            vp_b[dst_b + 4] = (float16_t)values_hi.x;
            vp_b[dst_b + 5] = (float16_t)values_hi.y;
            vp_b[dst_b + 6] = (float16_t)values_hi.z;
            vp_b[dst_b + 7] = (float16_t)values_hi.w;
#endif
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint k = 0; k < VP_BK; k += VP_TK) {
            VpB b = VpB::Load(vp_b, warp_c * VP_WN * VP_PITCH + k, VP_PITCH, MatrixLayout::ColMajor);
            VpA a = VpA::Load(vp_a, warp_r * VP_WM * VP_PITCH + k, VP_PITCH, MatrixLayout::RowMajor);
            sum.MultiplyAccumulate(a, b);
        }
        GroupMemoryBarrierWithGroupSync();
#if !VP_FULL_F16 && !VP_NATIVE_F32
        if ((k0 + VP_BK) % 64 == 0) {
            [unroll] for (uint e = 0; e < VP_ELEMS; ++e) {
                totals[e] += (float)sum.Get(e);
            }
            sum = VpC::Splat((float16_t)0);
        }
#endif
    }

    const uint dr = row + warp_r * VP_WM;
    const uint dc = col + warp_c * VP_WN;
#if VP_LDS_OUTPUT
    // Share one output tile to stay below the threadgroup LDS limit.
    for (uint owner = 0; owner < VP_THREADS / WAVE_SIZE; ++owner) {
        if (wave == owner) {
            sum.Store(vp_c, 0, VP_WN, MatrixLayout::RowMajor);
        }
        GroupMemoryBarrierWithGroupSync();
        if (wave == owner) {
            for (uint e = tid % WAVE_SIZE; e < VP_WM * VP_WN; e += WAVE_SIZE) {
                dst.Store(d_base + (dc + e % VP_WN) * nb1 + (dr + e / VP_WN) * 4, asuint(vp_c[e]));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
#elif VP_NATIVE_F32
    sum.Store(dst, d_base + dc * nb1 + dr * 4, nb1, MatrixLayout::ColMajor, 4);
#elif VP_FULL_F16
    [unroll] for (uint e = 0; e < VP_ELEMS; ++e) {
        sum.Set(e, clamp(sum.Get(e), (float16_t)-65504.0f, (float16_t)65504.0f));
    }
    VpOut value = sum.Cast<ComponentType::F32>();
    value.Store(dst, d_base + dc * nb1 + dr * 4, nb1, MatrixLayout::ColMajor, 4);
#else
    [unroll] for (uint e = 0; e < VP_ELEMS; ++e) {
        const uint2 rc = sum.GetCoordinate(e);
        dst.Store(d_base + (dc + rc.y) * nb1 + (dr + rc.x) * 4, asuint(totals[e]));
    }
#endif
}
