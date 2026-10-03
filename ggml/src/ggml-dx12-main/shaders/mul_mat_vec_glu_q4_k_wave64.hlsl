#include "ggml_common.hlsli"

float4 unpack_q4_bytes(uint packed) {
    return float4(packed & 0xFFu, (packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, packed >> 24);
}

float q4_dot4(float4 q, float4 x) {
    return mad(q.x, x.x, mad(q.y, x.y, mad(q.z, x.z, q.w * x.w)));
}

float q4_block_dot(ByteAddressBuffer weights, uint off, uint half_block, uint q_offset,
                   float4 x0, float4 x1, float4 x2, float4 x3, float4 sums) {
    uint4 header = weights.Load4(off);
    float d = f16_to_f32(header.x & 0xFFFFu);
    float m = f16_to_f32(header.x >> 16);
    uint shift = half_block * 16u;
    uint s0 = (header.y >> shift) & 0xFFFFu;
    uint s4 = (header.z >> shift) & 0xFFFFu;
    uint s8 = (header.w >> shift) & 0xFFFFu;
    uint low = s0 | (s4 << 16);
    uint high = (((s8 << 12) | s8) & 0x0F0F0F0Fu) | ((low & 0xC0C0C0C0u) >> 2);
    float4 low_scales = unpack_q4_bytes(low & 0x3F3F3F3Fu);
    float4 high_scales = unpack_q4_bytes(high);
    float4 scales = d * float4(low_scales.xy, high_scales.xy);
    float4 mins = m * float4(low_scales.zw, high_scales.zw);

    uint q0 = weights.Load(off + 16u + q_offset);
    uint q1 = weights.Load(off + 80u + q_offset);
    float sx = q4_dot4(unpack_q4_bytes(q0 & 0x0F0F0F0Fu), x0);
    float sy = q4_dot4(unpack_q4_bytes((q0 >> 4) & 0x0F0F0F0Fu), x1);
    float sz = q4_dot4(unpack_q4_bytes(q1 & 0x0F0F0F0Fu), x2);
    float sw = q4_dot4(unpack_q4_bytes((q1 >> 4) & 0x0F0F0F0Fu), x3);
    float smin = mins.x * sums.x + mins.y * sums.y + mins.z * sums.z + mins.w * sums.w;
    return (sx * scales.x + sy * scales.y + sz * scales.z + sw * scales.w) - smin;
}

// Same gate/up and optional RMS ABI as mul_mat_vec_glu_q4_k.
[WaveSize(64)]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2u;
    if (row0 >= ne0) {
        return;
    }
    uint row1 = min(row0 + 1u, ne0 - 1u);
    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;
    uint weight_batch = (i2 * ne02 / ne2) * nb02 + (i3 * ne03 / ne3) * nb03;
    uint gate0 = src0_offset + weight_batch + row0 * nb01;
    uint gate1 = src0_offset + weight_batch + row1 * nb01;
    uint up0 = op1 + weight_batch + row0 * nb01;
    uint up1 = op1 + weight_batch + row1 * nb01;
    uint input_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint half_block = (tid & 15u) >> 3;
    uint lane_offset = (tid & 7u) * 4u;
    uint q_offset = half_block * 32u + lane_offset;
    uint y_offset = half_block * 64u + lane_offset;
    float4 acc = 0.0f;
#if RMS_FUSED
    float ss = 0.0f;
#endif

    for (uint block = tid >> 4; block < ne00 / 256u; block += 4u) {
        uint k = block * 256u + y_offset;
        uint off = input_base + k * 4u;
        float4 x0 = asfloat(src1.Load4(off));
        float4 x1 = asfloat(src1.Load4(off + 128u));
        float4 x2 = asfloat(src1.Load4(off + 512u));
        float4 x3 = asfloat(src1.Load4(off + 640u));
#if RMS_FUSED
        ss += x0.x*x0.x + x0.y*x0.y + x0.z*x0.z + x0.w*x0.w
            + x1.x*x1.x + x1.y*x1.y + x1.z*x1.z + x1.w*x1.w
            + x2.x*x2.x + x2.y*x2.y + x2.z*x2.z + x2.w*x2.w
            + x3.x*x3.x + x3.y*x3.y + x3.z*x3.z + x3.w*x3.w;
        x0 *= asfloat(src6.Load4(k * 4u));
        x1 *= asfloat(src6.Load4(k * 4u + 128u));
        x2 *= asfloat(src6.Load4(k * 4u + 512u));
        x3 *= asfloat(src6.Load4(k * 4u + 640u));
#endif
        float4 sums = float4(x0.x + x0.y + x0.z + x0.w,
                             x1.x + x1.y + x1.z + x1.w,
                             x2.x + x2.y + x2.z + x2.w,
                             x3.x + x3.y + x3.z + x3.w);
        uint block_off = block * 144u;
        acc.x += q4_block_dot(src0, gate0 + block_off, half_block, q_offset, x0, x1, x2, x3, sums);
        acc.y += q4_block_dot(src0, gate1 + block_off, half_block, q_offset, x0, x1, x2, x3, sums);
        acc.z += q4_block_dot(src2, up0 + block_off, half_block, q_offset, x0, x1, x2, x3, sums);
        acc.w += q4_block_dot(src2, up1 + block_off, half_block, q_offset, x0, x1, x2, x3, sums);
    }

    float4 result = WaveActiveSum(acc);
#if RMS_FUSED
    float rms_scale = 1.0f / sqrt(WaveActiveSum(ss) / (float)ne00 + asfloat(op14));
    result *= rms_scale;
#endif
    if (tid == 0u) {
        float y0 = (result.x / (1.0f + exp(-result.x))) * result.z;
        store_auto(dst, offset_4d(row0, 0u, i2, i3, nb0, nb1, nb2, nb3, dst_offset), y0, dst_esize);
        if (row0 + 1u < ne0) {
            float y1 = (result.y / (1.0f + exp(-result.y))) * result.w;
            store_auto(dst, offset_4d(row0 + 1u, 0u, i2, i3, nb0, nb1, nb2, nb3, dst_offset), y1, dst_esize);
        }
    }
}
