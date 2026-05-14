// mul_mat_vec_q8_0_dp4a.hlsl - dp4a-accelerated Q8_0 matvec (M=1)
//
// Replaces the scalar float-dequant Q8_0 matvec with packed int8 dot products.
// Q8_0 is pure scale * int8 (no min term, no precision drift), so this is
// safe on NVIDIA (unlike Q4_K dp4a matvec which is gated off).
//
// 32 threads cooperate per workgroup: 8 threads per Q8_0 block × 4 blocks per
// iteration. Each thread loads one packed uint32 (4 int8 weights) and one
// packed uint32 (4 int8 Q8_1 activations) and issues one dot4add_i8packed.
// Two output rows share activation loads.
//
// Q8_0 block (34 bytes): d(f16) + qs[32] (int8 packed as 8 x uint32, but the
//                        qs region starts at byte offset 2 from the block,
//                        which is 2-byte aligned but NOT 4-byte aligned).
// Q8_1 block (36 bytes): ds(2xf16 packed) + qs[32] (8 x uint32, 4-byte aligned).
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  32
#endif
#define QK8_0       32
#define Q8_0_BSIZE  34
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

// Two rows × up to 4 waves: 8 entries is enough for any GROUP_SIZE up to 256
// when the wave size is at least the smallest variant we ship (16).
groupshared float shared_acc[2 * 8];

// Q8_0 qs starts 2 bytes into the block (after the f16 scale), so qs+l0 is
// 2-byte aligned but not 4-byte aligned. Load two adjacent words and shift.
uint read_u32_q80(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_q80(ByteAddressBuffer buf, uint byte_off) {
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
    uint num_blocks = K / QK8_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    // Q8_1 input is laid out flat in the scratch buffer: row i1 at
    // ((i3*ne12+i2)*ne11+i1)*num_blocks*Q8_1_BSIZE. M=1 here so i1=0.
    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_blocks * Q8_1_BSIZE;

    // 8 threads per block, BLOCKS_PER_ITER blocks per iteration
    uint sub  = tid / 8;          // 0..(BLOCKS_PER_ITER-1)
    uint lane = tid % 8;          // 0..7 (which uint32 within the block)
    uint l0   = lane * 4;         // byte offset within qs[]

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            // Q8_1 activation block (shared across both rows)
            uint q8_off = q8_vec_base + block_idx * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float a_d = f16_to_f32(ds & 0xFFFFu);
            uint a_packed = src1.Load(q8_off + 4 + l0);

            // Row 0
            {
                uint w_off = src0_row0 + block_idx * Q8_0_BSIZE;
                float w_d = read_f16_q80(src0, w_off);
                uint w_packed = read_u32_q80(src0, w_off + 2 + l0);

                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc0 += w_d * a_d * float(isum);
            }

            // Row 1
            {
                uint w_off = src0_row1 + block_idx * Q8_0_BSIZE;
                float w_d = read_f16_q80(src0, w_off);
                uint w_packed = read_u32_q80(src0, w_off + 2 + l0);

                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc1 += w_d * a_d * float(isum);
            }
        }
    }

    // Reduction: wave intrinsic across full wave, then shared-memory tree
    // for the (rare) case num_waves > 1.
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);

    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
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
