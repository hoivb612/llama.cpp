// mul_mat_vec_q5k_mr.hlsl - Multi-row Q5_K matvec (2 rows per group)
//
// Processes 2 output rows per thread group, sharing activation loads.
// Halves activation memory traffic — critical on UMA bandwidth.
//
// Dispatch: groups_x = ceil(N/2), groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q5K_BSIZE   176
#define N_ROWS      2

groupshared float shared_acc[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row_base = (group_id.y * 65535u + group_id.x) * N_ROWS;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (row_base >= ne0) return;
    bool valid_row1 = (row_base + 1 < ne0);

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_row0 = src0_offset + row_base * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row1 = src0_row0 + nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint il = itid / 4;
    uint ir = itid % 4;
    uint v_im = il / 2;
    uint v_in = il % 2;

    uint l0 = 4 * ir + 2 * v_in;
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 64 * v_im + l0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // === Load activations ONCE (shared between both rows) ===
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

        // Pre-compute activation sums for min compensation (shared)
        float sum_by0123 = by0 + by1 + by2 + by3;
        float sum_by4567 = by4 + by5 + by6 + by7;
        float sum_by89ab = by8 + by9 + by10 + by11;
        float sum_bycdef = by12 + by13 + by14 + by15;

        // === Row 0 ===
        {
            uint block_off = src0_row0 + block_idx * Q5K_BSIZE;

            uint dm_raw = src0.Load(block_off);
            float dall = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

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

            // Load qs + qh and decode Q5_K values (4-byte aligned loads)
            uint qs_off = block_off + 48;
            uint qs_byte0  = qs_off + q_offset;
            uint qs_byte16 = qs_off + q_offset + 16;
            uint qs_raw0   = src0.Load(qs_byte0 & ~3u);
            uint qs_raw16  = src0.Load(qs_byte16 & ~3u);
            uint qs_lo0  = (qs_byte0  & 2u) ? (qs_raw0  >> 16) : (qs_raw0  & 0xFFFFu);
            uint qs_lo16 = (qs_byte16 & 2u) ? (qs_raw16 >> 16) : (qs_raw16 & 0xFFFFu);
            uint qs0_16_u32 = qs_lo0 | (qs_lo16 << 16);

            uint qs_byte64 = qs_off + q_offset + 64;
            uint qs_byte80 = qs_off + q_offset + 80;
            uint qs_raw64  = src0.Load(qs_byte64 & ~3u);
            uint qs_raw80  = src0.Load(qs_byte80 & ~3u);
            uint qs_lo64 = (qs_byte64 & 2u) ? (qs_raw64 >> 16) : (qs_raw64 & 0xFFFFu);
            uint qs_lo80 = (qs_byte80 & 2u) ? (qs_raw80 >> 16) : (qs_raw80 & 0xFFFFu);
            uint qs64_80_u32 = qs_lo64 | (qs_lo80 << 16);

            uint qs0_lo4 = qs0_16_u32 & 0x0F0F0F0Fu;
            uint qs0_hi4 = (qs0_16_u32 >> 4) & 0x0F0F0F0Fu;
            uint qs64_lo4 = qs64_80_u32 & 0x0F0F0F0Fu;
            uint qs64_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0Fu;

            // qh high bits (4-byte aligned loads)
            uint qh_off = block_off + 16;
            uint qh_byte0  = qh_off + l0;
            uint qh_byte16 = qh_off + l0 + 16;
            uint qh_raw0   = src0.Load(qh_byte0 & ~3u);
            uint qh_raw16  = src0.Load(qh_byte16 & ~3u);
            uint qh_lo0  = (qh_byte0  & 2u) ? (qh_raw0  >> 16) : (qh_raw0  & 0xFFFFu);
            uint qh_lo16 = (qh_byte16 & 2u) ? (qh_raw16 >> 16) : (qh_raw16 & 0xFFFFu);
            uint qh = qh_lo0 | (qh_lo16 << 16);

            uint qh_shift = 2 * v_im;
            qs0_lo4  += ((qh >> qh_shift) & 0x01010101u) << 4;
            qs0_hi4  += ((qh >> qh_shift) & 0x02020202u) << 3;
            qs64_lo4 += (qh >> qh_shift) & 0x10101010u;
            qs64_hi4 += ((qh >> qh_shift) & 0x20202020u) >> 1;

            float q0  = float((qs0_lo4 >>  0) & 0xFFu);  float q1  = float((qs0_lo4 >>  8) & 0xFFu);
            float q2  = float((qs0_lo4 >> 16) & 0xFFu);  float q3  = float((qs0_lo4 >> 24) & 0xFFu);
            float q4  = float((qs0_hi4 >>  0) & 0xFFu);  float q5  = float((qs0_hi4 >>  8) & 0xFFu);
            float q6  = float((qs0_hi4 >> 16) & 0xFFu);  float q7  = float((qs0_hi4 >> 24) & 0xFFu);
            float q8  = float((qs64_lo4 >>  0) & 0xFFu); float q9  = float((qs64_lo4 >>  8) & 0xFFu);
            float q10 = float((qs64_lo4 >> 16) & 0xFFu); float q11 = float((qs64_lo4 >> 24) & 0xFFu);
            float q12 = float((qs64_hi4 >>  0) & 0xFFu); float q13 = float((qs64_hi4 >>  8) & 0xFFu);
            float q14 = float((qs64_hi4 >> 16) & 0xFFu); float q15 = float((qs64_hi4 >> 24) & 0xFFu);

            float sx = mad(q0, by0, mad(q1, by1, mad(q2, by2, q3*by3)));
            float sy = mad(q4, by4, mad(q5, by5, mad(q6, by6, q7*by7)));
            float sz = mad(q8, by8, mad(q9, by9, mad(q10, by10, q11*by11)));
            float sw = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15*by15)));

            float smin = mad(sc2, sum_by0123, mad(sc3, sum_by4567,
                         mad(sc6, sum_by89ab, sc7*sum_bycdef)));

            acc0 += dall * mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw*sc5))) - dmin * smin;
        }

        // === Row 1 (reuse activations) ===
        if (valid_row1) {
            uint block_off = src0_row1 + block_idx * Q5K_BSIZE;

            uint dm_raw = src0.Load(block_off);
            float dall = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

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

            // Load qs + qh and decode Q5_K values (4-byte aligned loads)
            uint qs_off = block_off + 48;
            uint qs_byte0  = qs_off + q_offset;
            uint qs_byte16 = qs_off + q_offset + 16;
            uint qs_raw0   = src0.Load(qs_byte0 & ~3u);
            uint qs_raw16  = src0.Load(qs_byte16 & ~3u);
            uint qs_lo0  = (qs_byte0  & 2u) ? (qs_raw0  >> 16) : (qs_raw0  & 0xFFFFu);
            uint qs_lo16 = (qs_byte16 & 2u) ? (qs_raw16 >> 16) : (qs_raw16 & 0xFFFFu);
            uint qs0_16_u32 = qs_lo0 | (qs_lo16 << 16);

            uint qs_byte64 = qs_off + q_offset + 64;
            uint qs_byte80 = qs_off + q_offset + 80;
            uint qs_raw64  = src0.Load(qs_byte64 & ~3u);
            uint qs_raw80  = src0.Load(qs_byte80 & ~3u);
            uint qs_lo64 = (qs_byte64 & 2u) ? (qs_raw64 >> 16) : (qs_raw64 & 0xFFFFu);
            uint qs_lo80 = (qs_byte80 & 2u) ? (qs_raw80 >> 16) : (qs_raw80 & 0xFFFFu);
            uint qs64_80_u32 = qs_lo64 | (qs_lo80 << 16);

            uint qs0_lo4 = qs0_16_u32 & 0x0F0F0F0Fu;
            uint qs0_hi4 = (qs0_16_u32 >> 4) & 0x0F0F0F0Fu;
            uint qs64_lo4 = qs64_80_u32 & 0x0F0F0F0Fu;
            uint qs64_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0Fu;

            // qh high bits (4-byte aligned loads)
            uint qh_off = block_off + 16;
            uint qh_byte0  = qh_off + l0;
            uint qh_byte16 = qh_off + l0 + 16;
            uint qh_raw0   = src0.Load(qh_byte0 & ~3u);
            uint qh_raw16  = src0.Load(qh_byte16 & ~3u);
            uint qh_lo0  = (qh_byte0  & 2u) ? (qh_raw0  >> 16) : (qh_raw0  & 0xFFFFu);
            uint qh_lo16 = (qh_byte16 & 2u) ? (qh_raw16 >> 16) : (qh_raw16 & 0xFFFFu);
            uint qh = qh_lo0 | (qh_lo16 << 16);

            uint qh_shift = 2 * v_im;
            qs0_lo4  += ((qh >> qh_shift) & 0x01010101u) << 4;
            qs0_hi4  += ((qh >> qh_shift) & 0x02020202u) << 3;
            qs64_lo4 += (qh >> qh_shift) & 0x10101010u;
            qs64_hi4 += ((qh >> qh_shift) & 0x20202020u) >> 1;

            float q0  = float((qs0_lo4 >>  0) & 0xFFu);  float q1  = float((qs0_lo4 >>  8) & 0xFFu);
            float q2  = float((qs0_lo4 >> 16) & 0xFFu);  float q3  = float((qs0_lo4 >> 24) & 0xFFu);
            float q4  = float((qs0_hi4 >>  0) & 0xFFu);  float q5  = float((qs0_hi4 >>  8) & 0xFFu);
            float q6  = float((qs0_hi4 >> 16) & 0xFFu);  float q7  = float((qs0_hi4 >> 24) & 0xFFu);
            float q8  = float((qs64_lo4 >>  0) & 0xFFu); float q9  = float((qs64_lo4 >>  8) & 0xFFu);
            float q10 = float((qs64_lo4 >> 16) & 0xFFu); float q11 = float((qs64_lo4 >> 24) & 0xFFu);
            float q12 = float((qs64_hi4 >>  0) & 0xFFu); float q13 = float((qs64_hi4 >>  8) & 0xFFu);
            float q14 = float((qs64_hi4 >> 16) & 0xFFu); float q15 = float((qs64_hi4 >> 24) & 0xFFu);

            float sx = mad(q0, by0, mad(q1, by1, mad(q2, by2, q3*by3)));
            float sy = mad(q4, by4, mad(q5, by5, mad(q6, by6, q7*by7)));
            float sz = mad(q8, by8, mad(q9, by9, mad(q10, by10, q11*by11)));
            float sw = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15*by15)));

            float smin = mad(sc2, sum_by0123, mad(sc3, sum_by4567,
                         mad(sc6, sum_by89ab, sc7*sum_bycdef)));

            acc1 += dall * mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw*sc5))) - dmin * smin;
        }
    }

    // Reduction for row 0
    float wave_sum = WaveActiveSum(acc0);
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
        if (op0 == 1u) result += asfloat(src2.Load(op1 + row_base * 4));
        uint off_d = offset_4d(row_base, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }

    // Reduction for row 1
    if (valid_row1) {
        wave_sum = WaveActiveSum(acc1);
        if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
        GroupMemoryBarrierWithGroupSync();

        if (tid < num_waves) {
            float v = shared_acc[tid];
            v = WaveActiveSum(v);
            if (tid == 0) shared_acc[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid == 0) {
            float result = shared_acc[0];
            if (op0 == 1u) result += asfloat(src2.Load(op1 + (row_base + 1) * 4));
            uint off_d = offset_4d(row_base + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d, result, dst_esize);
        }
    }
}
