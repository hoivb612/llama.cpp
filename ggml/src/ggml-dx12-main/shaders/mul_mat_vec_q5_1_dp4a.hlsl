// mul_mat_vec_q5_1_dp4a.hlsl - dp4a-accelerated Q5_1 matvec (M=1)
//
// Q5_1 block (24 bytes, 4-byte aligned):
//   offset 0..1   : d (f16)
//   offset 2..3   : m (f16)
//   offset 4..7   : qh[4] (5th bits)
//   offset 8..23  : qs[16] (low 4 bits per element; qs[l] low nibble = elem l, high nibble = elem l+16)
//
// Element l = scale_w * (qs nibble | (qh_bit << 4)) + m,  unsigned q5 in [0,31]
//
// dot per block = scale_w * scale_a * dp4a_sum + m * sum(a_float)
//                                              = scale_w * scale_a * dp4a_sum + m * a_s
// where a_s = d_a * sum(a_int8) is the Q8_1 's' field.
//
// 32 threads, 8 threads/block × 4 blocks/iter, 2 rows/group sharing activations.
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  32
#endif
#define QK5_1       32
#define Q5_1_BSIZE  24
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

groupshared float shared_acc[2 * 8];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * NUM_ROWS;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK5_1;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_blocks * Q8_1_BSIZE;

    uint sub  = tid / 8;
    uint lane = tid % 8;
    uint half_sel  = lane / 4;
    uint l_in_half = (lane & 3u) * 4u;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            uint q8_off = q8_vec_base + block_idx * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float a_d = f16_to_f32(ds & 0xFFFFu);
            float a_s = f16_to_f32(ds >> 16);
            uint a_packed = src1.Load(q8_off + 4 + lane * 4);

            // --- Row 0 ---
            {
                uint w_off = src0_row0 + block_idx * Q5_1_BSIZE;
                uint dm_word = src0.Load(w_off);
                float w_d = f16_to_f32(dm_word & 0xFFFFu);
                float w_m = f16_to_f32(dm_word >> 16);
                uint qh_word = src0.Load(w_off + 4);
                uint qs4 = src0.Load(w_off + 8 + l_in_half);
                uint nibbles = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                uint qh4 = (qh_word >> (lane * 4u)) & 0xFu;
                uint qh_packed = ( (qh4 & 1u)        << 4)  |
                                 (((qh4 >> 1) & 1u) << 12) |
                                 (((qh4 >> 2) & 1u) << 20) |
                                 (((qh4 >> 3) & 1u) << 28);
                uint w_packed = nibbles | qh_packed;     // each byte in [0,31]
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                // m * a_s is per-block, but 8 lanes contribute to each block's
                // dp4a, so split it 1/8 per lane.
                acc0 += w_d * a_d * float(isum) + 0.125f * w_m * a_s;
            }

            // --- Row 1 ---
            {
                uint w_off = src0_row1 + block_idx * Q5_1_BSIZE;
                uint dm_word = src0.Load(w_off);
                float w_d = f16_to_f32(dm_word & 0xFFFFu);
                float w_m = f16_to_f32(dm_word >> 16);
                uint qh_word = src0.Load(w_off + 4);
                uint qs4 = src0.Load(w_off + 8 + l_in_half);
                uint nibbles = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                uint qh4 = (qh_word >> (lane * 4u)) & 0xFu;
                uint qh_packed = ( (qh4 & 1u)        << 4)  |
                                 (((qh4 >> 1) & 1u) << 12) |
                                 (((qh4 >> 2) & 1u) << 20) |
                                 (((qh4 >> 3) & 1u) << 28);
                uint w_packed = nibbles | qh_packed;
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc1 += w_d * a_d * float(isum) + 0.125f * w_m * a_s;
            }
        }
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);

    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]     = wave_sum0;
        shared_acc[8 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result0 = shared_acc[0];
        for (uint w = 1; w < num_waves; w++) result0 += shared_acc[w];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[8];
            for (uint w = 1; w < num_waves; w++) result1 += shared_acc[8 + w];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
