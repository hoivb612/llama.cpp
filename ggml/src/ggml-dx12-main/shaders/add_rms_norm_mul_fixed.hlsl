#include "ggml_common.hlsli"

groupshared float wave_sums[8];

WAVE_SIZE_ATTR
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i3 = gid.x / (ne1 * ne2);
    uint i2 = (gid.x / ne1) % ne2;
    uint i1 = gid.x % ne1;
    uint a_base = src0_offset + i1 * nb01 + i2 * nb02 + i3 * nb03;
    uint b_base = src1_offset + (i1 % ne11) * nb11 + (i2 % ne12) * nb12 + (i3 % ne13) * nb13;
#ifndef RMS_ONLY
    uint w_base = op1 + (i1 % op7) * op4 + (i2 % op8) * op5 + (i3 % op9) * op6;
    uint add_base = op0 + i1 * nb01 + i2 * nb02 + i3 * nb03;
#endif
    uint out_base = dst_offset + i1 * nb1 + i2 * nb2 + i3 * nb3;
    float4 values[4];
    precise float sum = 0.0f;
    [unroll]
    for (uint j = 0; j < 4; ++j) {
        uint offset = tid.x * 16 + j * 1024;
        values[j] = asfloat(src0.Load4(a_base + offset));
#ifndef RMS_ONLY
        values[j] += asfloat(src1.Load4(b_base + offset));
        dst.Store4(add_base + offset, asuint(values[j]));
#endif
        sum += values[j].x * values[j].x;
        sum += values[j].y * values[j].y;
        sum += values[j].z * values[j].z;
        sum += values[j].w * values[j].w;
    }
    float total = WaveActiveSum(sum);
#if WAVE_SIZE != 64
    uint wave = tid.x / WaveGetLaneCount();
    if (WaveIsFirstLane()) wave_sums[wave] = total;
    GroupMemoryBarrierWithGroupSync();
    total = 0.0f;
    for (uint w = 0; w < 64 / WaveGetLaneCount(); ++w) total += wave_sums[w];
#endif
#ifdef RMS_ONLY
    float scale = rsqrt(total / 1024.0f + asfloat(op0));
#else
    float scale = rsqrt(total / 1024.0f + asfloat(op2));
#endif
    [unroll]
    for (uint j = 0; j < 4; ++j) {
        uint offset = tid.x * 16 + j * 1024;
#ifdef RMS_ONLY
        float4 weight = asfloat(src1.Load4(b_base + offset));
#else
        float4 weight = asfloat(src2.Load4(w_base + offset));
#endif
        dst.Store4(out_base + offset, asuint(values[j] * scale * weight));
    }
}
