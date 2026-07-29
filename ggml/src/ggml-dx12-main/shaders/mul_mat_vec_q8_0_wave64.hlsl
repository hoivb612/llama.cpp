// One Q8_0 output row per wave64 with eight values per lane.

#include "ggml_common.hlsli"

#define GROUP_SIZE       64
#define QK8_0            32
#define Q8_0_BSIZE       34
#define VALUES_PER_LANE   8

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[GROUP_SIZE / WAVE_SIZE];
#endif

uint read_u32_unaligned(uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = src0.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = src0.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_unaligned(uint byte_offset) {
    uint word = src0.Load(byte_offset & ~3u);
    return f16_to_f32((word >> ((byte_offset & 2u) * 8u)) & 0xffffu);
}

float dot_q8_0(uint packed, float4 x) {
    int4 q = int4(
        (int)(packed << 24) >> 24,
        (int)(packed << 16) >> 24,
        (int)(packed <<  8) >> 24,
        (int)packed >> 24);
    return dot(float4(q), x);
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row = group_x_2d(group_id);
    if (row >= ne0) {
        return;
    }

    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + row * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_row = src1_offset + i2 * nb12 + i3 * nb13;

    float acc = 0.0f;
    for (uint k = local_id * VALUES_PER_LANE; k < ne00;
         k += GROUP_SIZE * VALUES_PER_LANE) {
        uint block = k / QK8_0;
        uint element = k & (QK8_0 - 1u);
        uint block_offset = src0_row + block * Q8_0_BSIZE;
        float d = read_f16_unaligned(block_offset);

        uint packed0 = read_u32_unaligned(block_offset + 2u + element);
        uint packed1 = read_u32_unaligned(block_offset + 6u + element);
        float4 x0 = asfloat(src1.Load4(src1_row + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_row + (k + 4u) * 4u));
        acc += d * (dot_q8_0(packed0, x0) + dot_q8_0(packed1, x1));
    }

    float sum = WaveActiveSum(acc);

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        wave_sums[local_id / WAVE_SIZE] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum = 0.0f;
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum += wave_sums[wave];
        }
    }
#endif

    if (local_id == 0u) {
        sum += load_fused_bias(row, i2, i3);
        uint dst_row = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, dst_row, sum, dst_esize);
    }
}
