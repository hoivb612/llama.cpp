// Two fused Q8_0 gate/up output rows per wave64, sharing activation loads.

#include "ggml_common.hlsli"

#define GROUP_SIZE 64
#define QK8_0 32
#define Q8_0_BSIZE 34
#define VALUES_PER_LANE 8

uint read_u32_unaligned(ByteAddressBuffer weights, uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = weights.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = weights.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_unaligned(ByteAddressBuffer weights, uint byte_offset) {
    uint word = weights.Load(byte_offset & ~3u);
    return f16_to_f32((word >> ((byte_offset & 2u) * 8u)) & 0xffffu);
}

float dot_q8_0(uint packed, float4 x) {
    int4 q = int4(
        (int)(packed << 24) >> 24,
        (int)(packed << 16) >> 24,
        (int)(packed << 8) >> 24,
        (int)packed >> 24);
    return dot(float4(q), x);
}

float q8_dot(ByteAddressBuffer weights, uint row_offset, uint block, uint element, float4 x0, float4 x1) {
    uint block_offset = row_offset + block * Q8_0_BSIZE;
    float d = read_f16_unaligned(weights, block_offset);
    uint packed0 = read_u32_unaligned(weights, block_offset + 2u + element);
    uint packed1 = read_u32_unaligned(weights, block_offset + 6u + element);
    return d * (dot_q8_0(packed0, x0) + dot_q8_0(packed1, x1));
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

    uint batch_offset = i2_src0 * nb02 + i3_src0 * nb03;
    uint gate_row0 = src0_offset + batch_offset + row0 * nb01;
    uint gate_row1 = src0_offset + batch_offset + row1 * nb01;
    uint up_row0 = op1 + batch_offset + row0 * nb01;
    uint up_row1 = op1 + batch_offset + row1 * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    for (uint k = local_id * VALUES_PER_LANE; k < ne00; k += GROUP_SIZE * VALUES_PER_LANE) {
        uint block = k / QK8_0;
        uint element = k & (QK8_0 - 1u);
        float4 x0 = asfloat(src1.Load4(src1_base + k * 4u));
        float4 x1 = asfloat(src1.Load4(src1_base + (k + 4u) * 4u));

        gate0 += q8_dot(src0, gate_row0, block, element, x0, x1);
        up0 += q8_dot(src2, up_row0, block, element, x0, x1);
        if (has_row1) {
            gate1 += q8_dot(src0, gate_row1, block, element, x0, x1);
            up1 += q8_dot(src2, up_row1, block, element, x0, x1);
        }
    }

    gate0 = WaveActiveSum(gate0);
    gate1 = WaveActiveSum(gate1);
    up0 = WaveActiveSum(up0);
    up1 = WaveActiveSum(up1);
    if (local_id == 0u) {
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint dst_row0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, dst_row0, result0, dst_esize);
        if (has_row1) {
            float result1 = (gate1 / (1.0f + exp(-gate1))) * up1;
            uint dst_row1 = offset_4d(row1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, dst_row1, result1, dst_esize);
        }
    }
}
