// mul_mat_vec_q4_0_dp4a.hlsl - dp4a-accelerated Q4_0 matvec (M=1)
//
// Q4_0 block (18 bytes, 2-byte aligned, NOT 4-byte aligned):
//   offset 0..1   : d (f16)
//   offset 2..17  : qs[16] (low 4 bits per element; qs[l] low nibble = elem l, high nibble = elem l+16)
//
// Element l = (qs[l%16] nibble) - 8, range [-8, 7]
//
// Trick: feed unsigned q4 values [0,15] to dp4a (fits in int8) and correct
// for the -8 bias afterwards using the Q8_1 's' field (= d_a * sum(a_int8)):
//   block_sum = w_d * (a_d * dp4a_sum - 8 * s)
// 8 lanes share each block's dp4a so each lane adds -8 * s / 8 = -s per block.
//
// Structurally identical to mul_mat_vec_q5_0_dp4a.hlsl minus the qh fifth-bit
// handling. 32 threads, 8 threads/block × 4 blocks/iter, 2 rows/group sharing
// activations. Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3
//
// NVIDIA policy: Q4_0 dp4a is safe (pure int8 dot, no min term, no FP16
// accumulator drift) — same reasoning as the Q5_0 / Q8_0 dp4a paths.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  32
#endif
#define QK4_0       32
#define Q4_0_BSIZE  18
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

groupshared float shared_acc[2 * 8];

uint read_u32_q4_0(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_q4_0(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
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
    uint num_blocks = K / QK4_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_blocks * Q8_1_BSIZE;

    uint sub  = tid / 8;          // 0..(BLOCKS_PER_ITER-1)
    uint lane = tid % 8;          // 0..7 (which 4-element chunk within block)
    uint half_sel  = lane / 4;    // 0 = low nibble half (elems 0..15), 1 = high (elems 16..31)
    uint l_in_half = (lane & 3u) * 4u;  // qs byte offset: 0,4,8,12

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            // Q8_1 activations (shared across both rows)
            uint q8_off = q8_vec_base + block_idx * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float a_d = f16_to_f32(ds & 0xFFFFu);
            float a_s = f16_to_f32(ds >> 16);          // d_a * sum(a_int8)
            // Q8_1 qs[lane*4 .. lane*4+3] for elems 0..3, 4..7, ..., 28..31
            uint a_packed = src1.Load(q8_off + 4 + lane * 4);

            // --- Row 0 ---
            {
                uint w_off = src0_row0 + block_idx * Q4_0_BSIZE;
                float w_d = read_f16_q4_0(src0, w_off);
                uint qs4 = read_u32_q4_0(src0, w_off + 2 + l_in_half);
                uint w_packed = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                // -8 * a_s correction is per-block, 8 lanes per block -> 1/8
                // per lane, so coefficient becomes 1.0 per lane.
                acc0 += w_d * (a_d * float(isum) - a_s);
            }

            // --- Row 1 ---
            {
                uint w_off = src0_row1 + block_idx * Q4_0_BSIZE;
                float w_d = read_f16_q4_0(src0, w_off);
                uint qs4 = read_u32_q4_0(src0, w_off + 2 + l_in_half);
                uint w_packed = (half_sel == 0u) ? (qs4 & 0x0F0F0F0Fu) : ((qs4 >> 4) & 0x0F0F0F0Fu);
                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc1 += w_d * (a_d * float(isum) - a_s);
            }
        }
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);

    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]      = wave_sum0;
        shared_acc[8 + wave_id]  = wave_sum1;
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
