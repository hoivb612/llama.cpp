// Two fused Q5_0 gate/up output rows per wave64 using Vulkan's lane mapping.
//
// RMS_FUSED variant (mul_mat_vec_glu_q5_0_vulkan_rows2_rms.hlsl): folds the
// preceding RMS_NORM+MUL into this matvec. src1 carries the pre-norm activation
// x and src6 the norm weight g; one pass accumulates the dots against x*g plus
// sum(x*x) and applies 1/rms once at the end. op14 = eps (float bits).

#include "ggml_common.hlsli"

#define GROUP_SIZE 64
#define QK5_0 32
#define Q5_0_BSIZE 22

uint read_u32_unaligned(ByteAddressBuffer buf, uint byte_offset) {
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

uint read_u16_unaligned(ByteAddressBuffer buf, uint byte_offset) {
    uint word = buf.Load(byte_offset & ~3u);
    return (word >> ((byte_offset & 2u) * 8u)) & 0xffffu;
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

float q5_dot(ByteAddressBuffer weights, uint row_offset, uint block, uint iqs, float4 x0, float4 x1) {
    uint block_offset = row_offset + block * Q5_0_BSIZE;
    float d = f16_to_f32(read_u16_unaligned(weights, block_offset));
    uint qh = read_u32_unaligned(weights, block_offset + 2u);
    uint packed0 = read_u16_unaligned(weights, block_offset + 6u + iqs);
    uint packed1 = read_u16_unaligned(weights, block_offset + 8u + iqs);
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
#if RMS_FUSED
    precise float acc_ss = 0.0f;
#endif
    for (uint col = local_id * 8u; col < ne00; col += GROUP_SIZE * 8u) {
        uint block_start = col & ~(QK5_0 - 1u);
        uint block = col / QK5_0;
        uint iqs = (col & (QK5_0 - 1u)) / 2u;
        float4 x_low = asfloat(src1.Load4(src1_base + (block_start + iqs) * 4u));
        float4 x_high = asfloat(src1.Load4(src1_base + (block_start + iqs + 16u) * 4u));
#if RMS_FUSED
        acc_ss += dot(x_low, x_low) + dot(x_high, x_high);
        x_low  *= asfloat(src6.Load4((block_start + iqs) * 4u));
        x_high *= asfloat(src6.Load4((block_start + iqs + 16u) * 4u));
#endif
        float4 x0 = float4(x_low.x, x_high.x, x_low.y, x_high.y);
        float4 x1 = float4(x_low.z, x_high.z, x_low.w, x_high.w);
        gate0 += q5_dot(src0, gate_row0, block, iqs, x0, x1);
        up0 += q5_dot(src2, up_row0, block, iqs, x0, x1);
        if (has_row1) {
            gate1 += q5_dot(src0, gate_row1, block, iqs, x0, x1);
            up1 += q5_dot(src2, up_row1, block, iqs, x0, x1);
        }
    }

    gate0 = WaveActiveSum(gate0);
    gate1 = WaveActiveSum(gate1);
    up0 = WaveActiveSum(up0);
    up1 = WaveActiveSum(up1);
#if RMS_FUSED
    float rms_scale = 1.0f / sqrt(WaveActiveSum(acc_ss) / (float)ne00 + asfloat(op14));
    gate0 *= rms_scale;
    gate1 *= rms_scale;
    up0   *= rms_scale;
    up1   *= rms_scale;
#endif
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
