// mul_mat_vec_q5_0_mr64.hlsl - dp4a Q5_0 matvec single-wave (AMD wave64).
//
// Variant of mul_mat_vec_q5_0_dp4a tuned for AMD wave64:
//   - GROUP_SIZE=64 fills exactly one wave (default 32-thread variant leaves
//     half the wave idle on AMD).
//   - NUM_ROWS=4 amortizes activation reads 4x and increases launches per
//     workgroup. Matches Q8_0 mr64 pattern.
//   - GROUP_SIZE == WARP_SIZE so the cross-lane reduction is a pure
//     WaveActiveSum with no shared memory or barrier.
//
// Q5_0 block (22 bytes, 2-byte aligned, NOT 4-byte aligned):
//   offset 0..1   : d (f16)
//   offset 2..5   : qh[4] (5th bits, one bit per element)
//   offset 6..21  : qs[16] (low/high nibbles for elems 0..31)
//
// Element l = (qs[l%16] nibble | (qh_bit << 4)) - 16, range [-16, 15]
// Trick: feed unsigned q5 [0,31] to dp4a and correct -16 via Q8_1 's' field.
//
// Dispatch: groups_x = (N+3)/4, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  64
#endif
#define QK5_0       32
#define Q5_0_BSIZE  22
#define Q8_1_BSIZE  36
#define NUM_ROWS    4
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

uint read_u32_q5_0(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_q5_0(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

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
    uint num_blocks = K / QK5_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + (row0 + 0) * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src0_row2 = src0_base + (row0 + 2) * nb01;
    uint src0_row3 = src0_base + (row0 + 3) * nb01;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_blocks * Q8_1_BSIZE;

    uint sub  = tid / 8;          // 0..(BLOCKS_PER_ITER-1)
    uint lane = tid % 8;          // 0..7 (which 4-element chunk within block)
    uint half_sel  = lane / 4;    // 0 = low nibble half, 1 = high
    uint l_in_half = (lane & 3u) * 4u;  // qs byte offset: 0,4,8,12

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;
    precise float acc2 = 0.0f;
    precise float acc3 = 0.0f;

    bool has_r1 = (row0 + 1 < ne0);
    bool has_r2 = (row0 + 2 < ne0);
    bool has_r3 = (row0 + 3 < ne0);

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            // Q8_1 activations (shared across all NUM_ROWS rows)
            uint q8_off = q8_vec_base + block_idx * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float a_d = f16_to_f32(ds & 0xFFFFu);
            float a_s = f16_to_f32(ds >> 16);          // d_a * sum(a_int8)
            uint a_packed = src1.Load(q8_off + 4 + lane * 4);

            // --- Row 0 ---
            {
                uint w_off = src0_row0 + block_idx * Q5_0_BSIZE;
                float w_d = read_f16_q5_0(src0, w_off);
                uint qh_word = read_u32_q5_0(src0, w_off + 2);
                uint qs4 = read_u32_q5_0(src0, w_off + 6 + l_in_half);
                uint nibbles = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                uint qh4 = (qh_word >> (lane * 4u)) & 0xFu;
                uint qh_packed = ( (qh4 & 1u)        << 4)  |
                                 (((qh4 >> 1) & 1u) << 12) |
                                 (((qh4 >> 2) & 1u) << 20) |
                                 (((qh4 >> 3) & 1u) << 28);
                uint w_packed = nibbles | qh_packed;
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc0 += w_d * (a_d * float(isum) - 2.0f * a_s);
            }

            // --- Row 1 ---
            if (has_r1) {
                uint w_off = src0_row1 + block_idx * Q5_0_BSIZE;
                float w_d = read_f16_q5_0(src0, w_off);
                uint qh_word = read_u32_q5_0(src0, w_off + 2);
                uint qs4 = read_u32_q5_0(src0, w_off + 6 + l_in_half);
                uint nibbles = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                uint qh4 = (qh_word >> (lane * 4u)) & 0xFu;
                uint qh_packed = ( (qh4 & 1u)        << 4)  |
                                 (((qh4 >> 1) & 1u) << 12) |
                                 (((qh4 >> 2) & 1u) << 20) |
                                 (((qh4 >> 3) & 1u) << 28);
                uint w_packed = nibbles | qh_packed;
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc1 += w_d * (a_d * float(isum) - 2.0f * a_s);
            }

            // --- Row 2 ---
            if (has_r2) {
                uint w_off = src0_row2 + block_idx * Q5_0_BSIZE;
                float w_d = read_f16_q5_0(src0, w_off);
                uint qh_word = read_u32_q5_0(src0, w_off + 2);
                uint qs4 = read_u32_q5_0(src0, w_off + 6 + l_in_half);
                uint nibbles = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                uint qh4 = (qh_word >> (lane * 4u)) & 0xFu;
                uint qh_packed = ( (qh4 & 1u)        << 4)  |
                                 (((qh4 >> 1) & 1u) << 12) |
                                 (((qh4 >> 2) & 1u) << 20) |
                                 (((qh4 >> 3) & 1u) << 28);
                uint w_packed = nibbles | qh_packed;
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc2 += w_d * (a_d * float(isum) - 2.0f * a_s);
            }

            // --- Row 3 ---
            if (has_r3) {
                uint w_off = src0_row3 + block_idx * Q5_0_BSIZE;
                float w_d = read_f16_q5_0(src0, w_off);
                uint qh_word = read_u32_q5_0(src0, w_off + 2);
                uint qs4 = read_u32_q5_0(src0, w_off + 6 + l_in_half);
                uint nibbles = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                uint qh4 = (qh_word >> (lane * 4u)) & 0xFu;
                uint qh_packed = ( (qh4 & 1u)        << 4)  |
                                 (((qh4 >> 1) & 1u) << 12) |
                                 (((qh4 >> 2) & 1u) << 20) |
                                 (((qh4 >> 3) & 1u) << 28);
                uint w_packed = nibbles | qh_packed;
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc3 += w_d * (a_d * float(isum) - 2.0f * a_s);
            }
        }
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    float wave_sum2 = WaveActiveSum(acc2);
    float wave_sum3 = WaveActiveSum(acc3);

    if (tid == 0) {
        float result0 = wave_sum0 + load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (has_r1) {
            float result1 = wave_sum1 + load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
        if (has_r2) {
            float result2 = wave_sum2 + load_fused_bias(row0 + 2, i2, i3);
            uint off_d2 = offset_4d(row0 + 2, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d2, result2, dst_esize);
        }
        if (has_r3) {
            float result3 = wave_sum3 + load_fused_bias(row0 + 3, i2, i3);
            uint off_d3 = offset_4d(row0 + 3, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d3, result3, dst_esize);
        }
    }
}
