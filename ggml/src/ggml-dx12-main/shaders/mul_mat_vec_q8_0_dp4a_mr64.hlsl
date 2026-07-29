// mul_mat_vec_q8_0_dp4a_mr64.hlsl - AMD wave64 Q8_0 dp4a matvec (M=1, 4 rows/group)
//
// Combines the single-wave 64-lane layout of mul_mat_vec_q8_0_mr64 (NUM_ROWS=4,
// no cross-wave sync, no shared memory) with dot4add_i8packed + Q8_1 activations
// from mul_mat_vec_q8_0_dp4a. The scalar mr64 variant does one fp mul+add per
// int8 weight; this variant collapses 4 int8 weights into one packed dp4a, so
// the K loop becomes ~4x cheaper in ALU. Targets the K<1536 working point on
// AMD wave64 where the existing scalar mr64 (flag=28) is currently the default.
//
// 8 threads cooperate per Q8_0 block (4 int8 weights per thread); GROUP_SIZE=64
// covers 8 blocks per iteration. Q8_1 activations (shared across the 4 rows)
// are loaded once per thread per block_iter.
//
// Dispatch: groups_x = (N+3)/4, groups_y = 1, groups_z = batch * ne2 * ne3.

#include "ggml_common.hlsli"

#define GROUP_SIZE       64
#define QK8_0            32
#define Q8_0_BSIZE       34
#define Q8_1_BSIZE       36
#define NUM_ROWS         4
#define BLOCKS_PER_ITER  (GROUP_SIZE / 8)   // 8 blocks per iter

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[NUM_ROWS][GROUP_SIZE / WAVE_SIZE];
#endif

// Q8_0 qs starts 2 bytes into the block (after the f16 scale), so qs+l0 is
// 2-byte aligned but not 4-byte aligned. Load two adjacent words and shift.
uint read_u32_q80_mr(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_q80_mr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

WAVE_SIZE_ATTR
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
    uint src0_rows[NUM_ROWS];
    [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
        src0_rows[r] = src0_base + (row0 + r) * nb01;
    }

    // Q8_1 input is laid out flat in the scratch buffer; M=1 so i1=0.
    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_blocks * Q8_1_BSIZE;

    // 8 threads per block, BLOCKS_PER_ITER blocks per iteration
    uint sub  = tid / 8;          // 0..(BLOCKS_PER_ITER-1)
    uint lane = tid % 8;          // 0..7 (which uint32 within the block)
    uint l0   = lane * 4;         // byte offset within qs[]

    precise float acc[NUM_ROWS];
    [unroll] for (uint i = 0; i < NUM_ROWS; i++) acc[i] = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            // Q8_1 activation block (shared across all 4 rows)
            uint q8_off = q8_vec_base + block_idx * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float a_d = f16_to_f32(ds & 0xFFFFu);
            uint a_packed = src1.Load(q8_off + 4 + l0);

            [unroll]
            for (uint r = 0; r < NUM_ROWS; r++) {
                uint w_off = src0_rows[r] + block_idx * Q8_0_BSIZE;
                float w_d = read_f16_q80_mr(src0, w_off);
                uint w_packed = read_u32_q80_mr(src0, w_off + 2 + l0);

                int isum = 0;
                isum = dot4add_i8packed(w_packed, a_packed, isum);
                acc[r] = mad(w_d * a_d, float(isum), acc[r]);
            }
        }
    }

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    [unroll]
    for (uint r = 0; r < NUM_ROWS; r++) {
        float s = WaveActiveSum(acc[r]);
        if (WaveIsFirstLane()) {
            wave_sums[r][tid / WAVE_SIZE] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        [unroll]
        for (uint r = 0; r < NUM_ROWS; r++) {
            if ((row0 + r) < ne0) {
                float s = 0.0f;
                [unroll]
                for (uint w = 0; w < GROUP_SIZE / WAVE_SIZE; w++) {
                    s += wave_sums[r][w];
                }
                s += load_fused_bias(row0 + r, i2, i3);
                uint off_d = offset_4d(row0 + r, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off_d, s, dst_esize);
            }
        }
    }
#else
    [unroll]
    for (uint r = 0; r < NUM_ROWS; r++) {
        float s = WaveActiveSum(acc[r]);
        if (tid == 0 && (row0 + r) < ne0) {
            s += load_fused_bias(row0 + r, i2, i3);
            uint off_d = offset_4d(row0 + r, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d, s, dst_esize);
        }
    }
#endif
}
