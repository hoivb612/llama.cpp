// Two Q5_0 output rows per wave64 using Vulkan's lane mapping.

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define GROUP_SIZE 64
#define QK5_0 32
#define Q5_0_BSIZE 22

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

uint read_u16_unaligned(uint byte_offset) {
    uint word = src0.Load(byte_offset & ~3u);
    return (word >> ((byte_offset & 2u) * 8u)) & 0xffffu;
}

float read_f16_unaligned(uint byte_offset) {
    return f16_to_f32(read_u16_unaligned(byte_offset));
}

float4 decode_q5_0_packed16(uint packed, uint qh, uint iqs) {
    uint h0 = ((qh >> iqs) << 4u) & 0x10u;
    uint h16 = (qh >> (iqs + 12u)) & 0x10u;
    uint h1 = ((qh >> (iqs + 1u)) << 4u) & 0x10u;
    uint h17 = (qh >> (iqs + 13u)) & 0x10u;
    return float4(
        (packed & 0x0fu) | h0,
        ((packed >> 4u) & 0x0fu) | h16,
        ((packed >> 8u) & 0x0fu) | h1,
        ((packed >> 12u) & 0x0fu) | h17) - 16.0f;
}

float q5_dot(uint row_offset, uint block, uint iqs, float4 x0, float4 x1) {
    uint block_offset = row_offset + block * Q5_0_BSIZE;
    float d = read_f16_unaligned(block_offset);
    uint qh = read_u32_unaligned(block_offset + 2u);
    uint packed0 = read_u16_unaligned(block_offset + 6u + iqs);
    uint packed1 = read_u16_unaligned(block_offset + 8u + iqs);
    float4 w0 = decode_q5_0_packed16(packed0, qh, iqs);
    float4 w1 = decode_q5_0_packed16(packed1, qh, iqs + 2u);
    return d * (dot(w0, x0) + dot(w1, x1));
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
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint col = local_id * 8u; col < ne00; col += GROUP_SIZE * 8u) {
        uint block_start = col & ~(QK5_0 - 1u);
        uint block = col / QK5_0;
        uint iqs = (col & (QK5_0 - 1u)) / 2u;
        float4 x_low = asfloat(src1.Load4(src1_base + (block_start + iqs) * 4u));
        float4 x_high = asfloat(src1.Load4(src1_base + (block_start + iqs + 16u) * 4u));
        float4 x0 = float4(x_low.x, x_high.x, x_low.y, x_high.y);
        float4 x1 = float4(x_low.z, x_high.z, x_low.w, x_high.w);
        acc0 += q5_dot(src0_row0, block, iqs, x0, x1);
        if (has_row1) {
            acc1 += q5_dot(src0_row1, block, iqs, x0, x1);
        }
    }

    float sum0 = WaveActiveSum(acc0);
    float sum1 = WaveActiveSum(acc1);
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
