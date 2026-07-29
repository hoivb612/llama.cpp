// mul_mat_vec_q8_0_dp4a_nc2.hlsl - Q8_0 dp4a matvec for small-M batch (M==2).
//
// Computes NUM_ROWS=2 output rows x NUM_COLS=2 output cols per workgroup,
// loading each Q8_0 weight block once and reusing the decoded weights across
// both activation columns. Closes the M=2..8 perf gap where the wmma tile
// path (fl=4) wastes 16x of its 32x32 tile.
//
// Q8_0 block (34 bytes): d(f16) + qs[32] (qs region 2-byte aligned).
// Q8_1 block (36 bytes): ds(2xf16 packed) + qs[32] (4-byte aligned).
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3
// Gated to ne11==2, ne12==1, ne13==1.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  32
#endif
#define QK8_0       32
#define Q8_0_BSIZE  34
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define NUM_COLS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

// Up to 4 waves x 4 accumulators (2 rows x 2 cols) = 16 entries.
groupshared float shared_acc[NUM_ROWS * NUM_COLS * 8];

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
    uint num_blocks = K / QK8_0;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    // Q8_1 base for this batch slot. ne11 cols follow contiguously.
    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_batch_base = src1_offset +
        ((i3_q8 * ne12 + i2_q8) * ne11) * num_blocks * Q8_1_BSIZE;
    uint q8_col0_base = q8_batch_base + 0u * num_blocks * Q8_1_BSIZE;
    uint q8_col1_base = q8_batch_base + 1u * num_blocks * Q8_1_BSIZE;

    uint sub  = tid / 8;          // 0..(BLOCKS_PER_ITER-1)
    uint lane = tid % 8;          // 0..7
    uint l0   = lane * 4;

    precise float acc00 = 0.0f;
    precise float acc01 = 0.0f;
    precise float acc10 = 0.0f;
    precise float acc11 = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            // Load Q8_0 weights ONCE per row, reused across both cols.
            uint w_off0 = src0_row0 + block_idx * Q8_0_BSIZE;
            float w_d0 = read_f16_q80(src0, w_off0);
            uint w_packed0 = read_u32_q80(src0, w_off0 + 2 + l0);

            uint w_off1 = src0_row1 + block_idx * Q8_0_BSIZE;
            float w_d1 = read_f16_q80(src0, w_off1);
            uint w_packed1 = read_u32_q80(src0, w_off1 + 2 + l0);

            // Col 0: load Q8_1 activation
            {
                uint q8_off = q8_col0_base + block_idx * Q8_1_BSIZE;
                uint ds = src1.Load(q8_off);
                float a_d = f16_to_f32(ds & 0xFFFFu);
                uint a_packed = src1.Load(q8_off + 4 + l0);

                int isum0 = 0;
                isum0 = dot4add_i8packed(w_packed0, a_packed, isum0);
                acc00 += w_d0 * a_d * float(isum0);

                int isum1 = 0;
                isum1 = dot4add_i8packed(w_packed1, a_packed, isum1);
                acc10 += w_d1 * a_d * float(isum1);
            }

            // Col 1: load Q8_1 activation
            {
                uint q8_off = q8_col1_base + block_idx * Q8_1_BSIZE;
                uint ds = src1.Load(q8_off);
                float a_d = f16_to_f32(ds & 0xFFFFu);
                uint a_packed = src1.Load(q8_off + 4 + l0);

                int isum0 = 0;
                isum0 = dot4add_i8packed(w_packed0, a_packed, isum0);
                acc01 += w_d0 * a_d * float(isum0);

                int isum1 = 0;
                isum1 = dot4add_i8packed(w_packed1, a_packed, isum1);
                acc11 += w_d1 * a_d * float(isum1);
            }
        }
    }

    // Wave reduce 4 accumulators
    float wave_sum00 = WaveActiveSum(acc00);
    float wave_sum01 = WaveActiveSum(acc01);
    float wave_sum10 = WaveActiveSum(acc10);
    float wave_sum11 = WaveActiveSum(acc11);

    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;

    if (WaveIsFirstLane()) {
        shared_acc[0 * 8 + wave_id] = wave_sum00;
        shared_acc[1 * 8 + wave_id] = wave_sum01;
        shared_acc[2 * 8 + wave_id] = wave_sum10;
        shared_acc[3 * 8 + wave_id] = wave_sum11;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float r00 = shared_acc[0 * 8];
        float r01 = shared_acc[1 * 8];
        float r10 = shared_acc[2 * 8];
        float r11 = shared_acc[3 * 8];
        for (uint w = 1; w < num_waves; w++) {
            r00 += shared_acc[0 * 8 + w];
            r01 += shared_acc[1 * 8 + w];
            r10 += shared_acc[2 * 8 + w];
            r11 += shared_acc[3 * 8 + w];
        }
        uint off00 = offset_4d(row0,     0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        uint off01 = offset_4d(row0,     1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off00, r00, dst_esize);
        store_auto(dst, off01, r01, dst_esize);

        if (row0 + 1 < ne0) {
            uint off10 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            uint off11 = offset_4d(row0 + 1, 1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off10, r10, dst_esize);
            store_auto(dst, off11, r11, dst_esize);
        }
    }
}
