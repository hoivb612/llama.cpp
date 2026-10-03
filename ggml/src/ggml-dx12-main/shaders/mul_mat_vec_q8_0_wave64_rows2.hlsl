// Two Q8_0 output rows per wave64, sharing activation loads.

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define GROUP_SIZE       64
#define QK8_0            32
#define Q8_0_BSIZE       34
#define VALUES_PER_LANE   8

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[2][GROUP_SIZE / WAVE_SIZE];
#endif

uint read_u32_unaligned(uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = src0.Load(aligned);
    uint hi = src0.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
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
    uint row0 = group_x_2d(group_id) * 2u;
    if (row0 >= ne0) {
        return;
    }
    uint row1 = row0 + 1u;
    bool has_row1 = row1 < ne0;

    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_batch = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_batch + row0 * nb01;
    uint src0_row1 = src0_batch + row1 * nb01;
    uint src1_row = src1_offset + i2 * nb12 + i3 * nb13;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint k = local_id * VALUES_PER_LANE; k < ne00;
         k += GROUP_SIZE * VALUES_PER_LANE) {
        uint block = k / QK8_0;
        uint element = k & (QK8_0 - 1u);
        uint block_offset0 = src0_row0 + block * Q8_0_BSIZE;
        uint block_offset1 = src0_row1 + block * Q8_0_BSIZE;
        float4 x0 = asfloat(src1.Load4(src1_row + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_row + (k + 4u) * 4u));

        float d0 = read_f16_unaligned(block_offset0);
        uint packed00 = read_u32_unaligned(block_offset0 + 2u + element);
        uint packed01 = read_u32_unaligned(block_offset0 + 6u + element);
        acc0 += d0 * (dot_q8_0(packed00, x0) + dot_q8_0(packed01, x1));

        if (has_row1) {
            float d1 = read_f16_unaligned(block_offset1);
            uint packed10 = read_u32_unaligned(block_offset1 + 2u + element);
            uint packed11 = read_u32_unaligned(block_offset1 + 6u + element);
            acc1 += d1 * (dot_q8_0(packed10, x0) + dot_q8_0(packed11, x1));
        }
    }

    float sum0 = WaveActiveSum(acc0);
    float sum1 = WaveActiveSum(acc1);

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        uint wave = local_id / WAVE_SIZE;
        wave_sums[0][wave] = sum0;
        wave_sums[1][wave] = sum1;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum0 = 0.0f;
        sum1 = 0.0f;
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum0 += wave_sums[0][wave];
            sum1 += wave_sums[1][wave];
        }
    }
#endif

    if (local_id == 0u) {
        sum0 += load_fused_bias(row0, i2, i3);
        if (has_row1) {
            sum1 += load_fused_bias(row1, i2, i3);
        }
        if (mmv_rope_active()) {
            // row0 = 2g, row1 = 2g+1 form a NORMAL-mode rotation pair.
            uint pair_in_head = (row0 % op10) / 2u;
            float out0, out1;
            mmv_rope_pair(pair_in_head, sum0, sum1, out0, out1);
            mmv_rope_store(row0, out0);
            if (has_row1) {
                mmv_rope_store(row1, out1);
            }
        } else if (mmv_scatter_active()) {
            mmv_store_scatter(row0, 0u, sum0);
            if (has_row1) {
                mmv_store_scatter(row1, 0u, sum1);
            }
        } else {
            uint dst_row0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, dst_row0, sum0, dst_esize);
            if (has_row1) {
                uint dst_row1 = offset_4d(row1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, dst_row1, sum1, dst_esize);
            }
        }
    }
}
