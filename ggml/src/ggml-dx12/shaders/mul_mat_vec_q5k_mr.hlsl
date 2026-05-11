// mul_mat_vec_q5k_mr.hlsl - Multi-row Q5_K matvec (M=1, 2 rows/group)
//
// Q5_K block (176 bytes): d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
// 256 threads, 16 per Q5_K block. Shares activation loads across 2 rows.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q5K_BSIZE   176

uint safe_load_u16(ByteAddressBuffer buf, uint byte_addr) {
    uint aligned = byte_addr & ~3u;
    uint raw = buf.Load(aligned);
    return ((byte_addr & 2u) != 0u) ? ((raw >> 16) & 0xFFFFu) : (raw & 0xFFFFu);
}

groupshared float shared_acc[64];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
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
        // --- Shared: load 16 activation values ---
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

        float by0  = asfloat(p01.x); float by1  = asfloat(p01.y);
        float by2  = asfloat(p23.x); float by3  = asfloat(p23.y);
        float by4  = asfloat(p45.x); float by5  = asfloat(p45.y);
        float by6  = asfloat(p67.x); float by7  = asfloat(p67.y);
        float by8  = asfloat(p89.x); float by9  = asfloat(p89.y);
        float by10 = asfloat(pab.x); float by11 = asfloat(pab.y);
        float by12 = asfloat(pcd.x); float by13 = asfloat(pcd.y);
        float by14 = asfloat(pef.x); float by15 = asfloat(pef.y);

        // --- Process both rows ---
        [unroll]
        for (uint r = 0; r < 2; r++) {
            uint block_off = (r == 0 ? src0_row0 : src0_row1) + block_idx * Q5K_BSIZE;

            uint dm_raw = src0.Load(block_off);
            float dall = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

            // Decode scales (same as Q4_K)
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

            // Load qs (Q5_K: qs at block_off + 48)
            uint qs_off = block_off + 48;
            uint qs_lo0 = safe_load_u16(src0, qs_off + q_offset);
            uint qs_lo16 = safe_load_u16(src0, qs_off + q_offset + 16);
            uint qs0_16_u32 = qs_lo0 | (qs_lo16 << 16);

            uint qs_lo64 = safe_load_u16(src0, qs_off + q_offset + 64);
            uint qs_lo80 = safe_load_u16(src0, qs_off + q_offset + 80);
            uint qs64_80_u32 = qs_lo64 | (qs_lo80 << 16);

            uint qs0_lo4 = qs0_16_u32 & 0x0F0F0F0Fu;
            uint qs0_hi4 = (qs0_16_u32 >> 4) & 0x0F0F0F0Fu;
            uint qs64_lo4 = qs64_80_u32 & 0x0F0F0F0Fu;
            uint qs64_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0Fu;

            // Load qh (Q5_K: qh at block_off + 16)
            uint qh_off = block_off + 16;
            uint qh_lo0 = safe_load_u16(src0, qh_off + l0);
            uint qh_lo16 = safe_load_u16(src0, qh_off + l0 + 16);
            uint qh = qh_lo0 | (qh_lo16 << 16);

            uint qh_shift = 2 * v_im;
            qs0_lo4  += ((qh >> qh_shift) & 0x01010101u) << 4;
            qs0_hi4  += ((qh >> qh_shift) & 0x02020202u) << 3;
            qs64_lo4 += (qh >> qh_shift) & 0x10101010u;
            qs64_hi4 += ((qh >> qh_shift) & 0x20202020u) >> 1;

            float q0  = float((qs0_lo4  >>  0) & 0xFFu); float q1  = float((qs0_lo4  >>  8) & 0xFFu);
            float q2  = float((qs0_lo4  >> 16) & 0xFFu); float q3  = float((qs0_lo4  >> 24) & 0xFFu);
            float q4  = float((qs0_hi4  >>  0) & 0xFFu); float q5  = float((qs0_hi4  >>  8) & 0xFFu);
            float q6  = float((qs0_hi4  >> 16) & 0xFFu); float q7  = float((qs0_hi4  >> 24) & 0xFFu);
            float q8  = float((qs64_lo4 >>  0) & 0xFFu); float q9  = float((qs64_lo4 >>  8) & 0xFFu);
            float q10 = float((qs64_lo4 >> 16) & 0xFFu); float q11 = float((qs64_lo4 >> 24) & 0xFFu);
            float q12 = float((qs64_hi4 >>  0) & 0xFFu); float q13 = float((qs64_hi4 >>  8) & 0xFFu);
            float q14 = float((qs64_hi4 >> 16) & 0xFFu); float q15 = float((qs64_hi4 >> 24) & 0xFFu);

            float sx = q0*by0 + q1*by1 + q2*by2 + q3*by3;
            float sy = q4*by4 + q5*by5 + q6*by6 + q7*by7;
            float sz = q8*by8 + q9*by9 + q10*by10 + q11*by11;
            float sw = q12*by12 + q13*by13 + q14*by14 + q15*by15;

            float smin = sc2*(by0+by1+by2+by3) + sc3*(by4+by5+by6+by7)
                       + sc6*(by8+by9+by10+by11) + sc7*(by12+by13+by14+by15);

            float row_acc = dall * (sx*sc0 + sy*sc1 + sz*sc4 + sw*sc5) - dmin * smin;
            if (r == 0) acc0 += row_acc;
            else        acc1 += row_acc;
        }
    }

    // Tree reduction
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
