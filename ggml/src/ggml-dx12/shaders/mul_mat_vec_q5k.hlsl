// mul_mat_vec_q5k.hlsl - Optimized matrix-vector multiply for Q5_K weights (M=1)
//
// Matches Vulkan mul_mat_vec_q5_k.comp approach:
// - 16 threads cooperate per Q5_K block  
// - Vectorized uint32 loads for qs + qh high-bit extraction
// - Packed scale decoding (same as Q4_K)
// - 256 threads / group, 1 output row / group
//
// Q5_K block layout (176 bytes): d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q5K_BSIZE   176

groupshared float shared_acc[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_id.x;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // 16 threads cooperate per Q5_K block
    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint il = itid / 4;
    uint ir = itid % 4;
    uint v_im = il / 2;
    uint v_in = il % 2;

    // Q5_K uses stride-2 element spacing (vs stride-4 for Q4_K)
    uint l0 = 4 * ir + 2 * v_in;        // 0,2,4,6,8,10,12,14
    uint q_offset = 32 * v_im + l0;     // qs byte offset
    uint y_offset = 64 * v_im + l0;     // activation element offset

    float acc = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        uint block_off = src0_row + block_idx * Q5K_BSIZE;

        // Load block header
        uint dm_raw = src0.Load(block_off);
        float dall = f16_to_f32(dm_raw & 0xFFFFu);
        float dmin = f16_to_f32(dm_raw >> 16);

        // Load and decode scales (identical to Q4_K)
        uint scales_off = block_off + 4;
        uint s_raw0, s_raw4, s_raw8;
        if (v_im == 0) {
            s_raw0 = src0.Load(scales_off) & 0xFFFFu;
            s_raw4 = src0.Load(scales_off + 4) & 0xFFFFu;
            s_raw8 = src0.Load(scales_off + 8) & 0xFFFFu;
        } else {
            s_raw0 = (src0.Load(scales_off) >> 16) & 0xFFFFu;
            s_raw4 = (src0.Load(scales_off + 4) >> 16) & 0xFFFFu;
            s_raw8 = (src0.Load(scales_off + 8) >> 16) & 0xFFFFu;
        }

        uint scale_0_4_l = (s_raw4 << 16) | s_raw0;
        uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;

        float sc0 = float((scale_0_4_l >>  0) & 0x3Fu);
        float sc1 = float((scale_0_4_l >>  8) & 0x3Fu);
        float sc2 = float((scale_0_4_l >> 16) & 0x3Fu);
        float sc3 = float((scale_0_4_l >> 24) & 0x3Fu);

        uint combined_8 = (((s_raw8 << 12) | s_raw8) & 0x0F0F0F0Fu) | scale_0_4_h;
        float sc4 = float((combined_8 >>  0) & 0xFFu);
        float sc5 = float((combined_8 >>  8) & 0xFFu);
        float sc6 = float((combined_8 >> 16) & 0xFFu);
        float sc7 = float((combined_8 >> 24) & 0xFFu);

        // Load qs as uint16 pairs combined into uint32
        // Q5_K: qs at block_off + 48 (after 4+12+32 = 48 bytes)
        uint qs_off = block_off + 48;
        // Load 2 uint16 from qs and combine (stride-2 access)
        uint qs_addr0 = qs_off + (q_offset & ~1u);
        uint qs_addr16 = qs_off + ((q_offset + 16) & ~1u);
        uint qs_raw0 = src0.Load(qs_addr0);
        uint qs_raw16 = src0.Load(qs_addr16);

        // Extract the right uint16 based on q_offset alignment
        uint qs_lo0, qs_lo16;
        if ((q_offset & 1u) == 0u) {
            qs_lo0 = qs_raw0 & 0xFFFFu;
            qs_lo16 = qs_raw16 & 0xFFFFu;
        } else {
            qs_lo0 = (qs_raw0 >> 8) & 0xFFFFu;  // shift by 8 bits for odd byte offset
            qs_lo16 = (qs_raw16 >> 8) & 0xFFFFu;
        }

        // Combine two uint16 into uint32 for each pair
        uint qs0_16_u32 = qs_lo0 | (qs_lo16 << 16);

        uint qs_addr64 = qs_off + ((q_offset + 64) & ~1u);
        uint qs_addr80 = qs_off + ((q_offset + 80) & ~1u);
        uint qs_raw64 = src0.Load(qs_addr64);
        uint qs_raw80 = src0.Load(qs_addr80);

        uint qs_lo64, qs_lo80;
        if ((q_offset & 1u) == 0u) {
            qs_lo64 = qs_raw64 & 0xFFFFu;
            qs_lo80 = qs_raw80 & 0xFFFFu;
        } else {
            qs_lo64 = (qs_raw64 >> 8) & 0xFFFFu;
            qs_lo80 = (qs_raw80 >> 8) & 0xFFFFu;
        }

        uint qs64_80_u32 = qs_lo64 | (qs_lo80 << 16);

        // Extract 4-bit values
        uint qs0_lo4 = qs0_16_u32 & 0x0F0F0F0Fu;
        uint qs0_hi4 = (qs0_16_u32 >> 4) & 0x0F0F0F0Fu;
        uint qs64_lo4 = qs64_80_u32 & 0x0F0F0F0Fu;
        uint qs64_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0Fu;

        // Load qh (high bits) - Q5_K: qh at block_off + 16 (after 4+12 = 16 bytes)
        uint qh_off = block_off + 16;
        // Load 4 bytes of qh as 2 × uint16 combined
        uint qh_addr0 = qh_off + (l0 & ~1u);
        uint qh_addr16 = qh_off + ((l0 + 16) & ~1u);
        uint qh_raw0 = src0.Load(qh_addr0);
        uint qh_raw16 = src0.Load(qh_addr16);

        uint qh_lo0, qh_lo16;
        if ((l0 & 1u) == 0u) {
            qh_lo0 = qh_raw0 & 0xFFFFu;
            qh_lo16 = qh_raw16 & 0xFFFFu;
        } else {
            qh_lo0 = (qh_raw0 >> 8) & 0xFFFFu;
            qh_lo16 = (qh_raw16 >> 8) & 0xFFFFu;
        }

        uint qh = qh_lo0 | (qh_lo16 << 16);

        // Extract high bits and add to 4-bit values to get 5-bit values
        uint qh_shift = 2 * v_im;
        qs0_lo4  += ((qh >> qh_shift) & 0x01010101u) << 4;
        qs0_hi4  += ((qh >> qh_shift) & 0x02020202u) << 3;
        qs64_lo4 += (qh >> qh_shift) & 0x10101010u;
        qs64_hi4 += ((qh >> qh_shift) & 0x20202020u) >> 1;

        // Unpack to floats
        float q0  = float((qs0_lo4 >>  0) & 0xFFu);
        float q1  = float((qs0_lo4 >>  8) & 0xFFu);
        float q2  = float((qs0_lo4 >> 16) & 0xFFu);
        float q3  = float((qs0_lo4 >> 24) & 0xFFu);
        float q4  = float((qs0_hi4 >>  0) & 0xFFu);
        float q5  = float((qs0_hi4 >>  8) & 0xFFu);
        float q6  = float((qs0_hi4 >> 16) & 0xFFu);
        float q7  = float((qs0_hi4 >> 24) & 0xFFu);
        float q8  = float((qs64_lo4 >>  0) & 0xFFu);
        float q9  = float((qs64_lo4 >>  8) & 0xFFu);
        float q10 = float((qs64_lo4 >> 16) & 0xFFu);
        float q11 = float((qs64_lo4 >> 24) & 0xFFu);
        float q12 = float((qs64_hi4 >>  0) & 0xFFu);
        float q13 = float((qs64_hi4 >>  8) & 0xFFu);
        float q14 = float((qs64_hi4 >> 16) & 0xFFu);
        float q15 = float((qs64_hi4 >> 24) & 0xFFu);

        // Load 16 activation values using paired Load2 (strided access, pairs 64 bytes apart)
        uint y1_off = src1_base + (block_idx * QK_K + y_offset) * 4;
        uint y2_off = y1_off + 128 * 4;

        uint2 p01 = src1.Load2(y1_off);
        uint2 p23 = src1.Load2(y1_off + 64);
        uint2 p45 = src1.Load2(y1_off + 128);
        uint2 p67 = src1.Load2(y1_off + 192);
        uint2 p89 = src1.Load2(y2_off);
        uint2 pab = src1.Load2(y2_off + 64);
        uint2 pcd = src1.Load2(y2_off + 128);
        uint2 pef = src1.Load2(y2_off + 192);

        float by0 = asfloat(p01.x); float by1 = asfloat(p01.y);
        float by2 = asfloat(p23.x); float by3 = asfloat(p23.y);
        float by4 = asfloat(p45.x); float by5 = asfloat(p45.y);
        float by6 = asfloat(p67.x); float by7 = asfloat(p67.y);
        float by8 = asfloat(p89.x); float by9 = asfloat(p89.y);
        float by10 = asfloat(pab.x); float by11 = asfloat(pab.y);
        float by12 = asfloat(pcd.x); float by13 = asfloat(pcd.y);
        float by14 = asfloat(pef.x); float by15 = asfloat(pef.y);

        // Compute partial dot products using mad() FMA chains
        float sx = mad(q0, by0, mad(q1, by1, mad(q2, by2, q3*by3)));
        float sy = mad(q4, by4, mad(q5, by5, mad(q6, by6, q7*by7)));
        float sz = mad(q8, by8, mad(q9, by9, mad(q10, by10, q11*by11)));
        float sw = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15*by15)));

        float smin = mad(sc2, by0+by1+by2+by3, mad(sc3, by4+by5+by6+by7,
                     mad(sc6, by8+by9+by10+by11, sc7*(by12+by13+by14+by15))));

        acc += dall * mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw*sc5))) - dmin * smin;
    }

    // Wave-intrinsic reduction (2 barriers instead of 8)
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WaveGetLaneCount();
    if (tid < num_waves) {
        float v = shared_acc[tid];
        v = WaveActiveSum(v);
        if (tid == 0) shared_acc[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result = shared_acc[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
