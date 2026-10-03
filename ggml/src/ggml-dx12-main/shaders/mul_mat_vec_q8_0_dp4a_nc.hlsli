// mul_mat_vec_q8_0_dp4a_nc.hlsli - Q8_0 dp4a matvec for small-M batches.
//
// Computes NUM_ROWS output rows x NUM_COLS output cols per workgroup, loading
// each Q8_0 weight block once and reusing the decoded weights across all
// activation columns. Closes the M=2..8 gap where the wmma tile path wastes
// most of its 32x32 tile. Wrappers set NUM_COLS.
//
// Q8_0 block (34 bytes): d(f16) + qs[32] (qs region 2-byte aligned).
// Q8_1 block (36 bytes): ds(2xf16 packed) + qs[32] (4-byte aligned).
//
// Dispatch: groups_x = ceil(N/NUM_ROWS), groups_y = 1, groups_z = batch*ne2*ne3
// Gated to ne11==NUM_COLS, ne12==1, ne13==1.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  32
#endif
#ifndef NUM_COLS
#define NUM_COLS    2
#endif
#define QK8_0       32
#define Q8_0_BSIZE  34
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define BLOCKS_PER_ITER (GROUP_SIZE / 8)

// Up to 8 waves per accumulator (NUM_ROWS x NUM_COLS).
groupshared float shared_acc[NUM_ROWS * NUM_COLS * 8];

uint read_u32_q80(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
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

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_batch_base = src1_offset +
        ((i3_q8 * ne12 + i2_q8) * ne11) * num_blocks * Q8_1_BSIZE;

    uint sub  = tid / 8;          // 0..(BLOCKS_PER_ITER-1)
    uint lane = tid % 8;          // 0..7
    uint l0   = lane * 4;

    precise float acc[NUM_ROWS][NUM_COLS];
    [unroll] for (uint rr = 0; rr < NUM_ROWS; rr++) {
        [unroll] for (uint cc = 0; cc < NUM_COLS; cc++) {
            acc[rr][cc] = 0.0f;
        }
    }

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block_idx = block_iter + sub;
        if (block_idx < num_blocks) {
            float w_d[NUM_ROWS];
            uint  w_packed[NUM_ROWS];
            [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
                uint w_off = src0_base + (row0 + r) * nb01 + block_idx * Q8_0_BSIZE;
                w_d[r]      = read_f16_q80(src0, w_off);
                w_packed[r] = read_u32_q80(src0, w_off + 2 + l0);
            }

            [unroll] for (uint c = 0; c < NUM_COLS; c++) {
                if (c >= ne11) break;
                uint q8_off = q8_batch_base + c * num_blocks * Q8_1_BSIZE +
                              block_idx * Q8_1_BSIZE;
                uint ds = src1.Load(q8_off);
                float a_d = f16_to_f32(ds & 0xFFFFu);
                uint a_packed = src1.Load(q8_off + 4 + l0);

                [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
                    int isum = 0;
                    isum = dot4add_i8packed(w_packed[r], a_packed, isum);
                    acc[r][c] += w_d[r] * a_d * float(isum);
                }
            }
        }
    }

    uint wave_id = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    if (num_waves == 0) num_waves = 1;

    [unroll] for (uint rr2 = 0; rr2 < NUM_ROWS; rr2++) {
        [unroll] for (uint cc2 = 0; cc2 < NUM_COLS; cc2++) {
            if (cc2 >= ne11) break;
            float ws = WaveActiveSum(acc[rr2][cc2]);
            if (WaveIsFirstLane()) {
                shared_acc[(rr2 * NUM_COLS + cc2) * 8 + wave_id] = ws;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        [unroll] for (uint rr3 = 0; rr3 < NUM_ROWS; rr3++) {
            if (row0 + rr3 >= ne0) continue;
            [unroll] for (uint cc3 = 0; cc3 < NUM_COLS; cc3++) {
                if (cc3 >= ne11) break;
                float r = shared_acc[(rr3 * NUM_COLS + cc3) * 8];
                for (uint w = 1; w < num_waves; w++) {
                    r += shared_acc[(rr3 * NUM_COLS + cc3) * 8 + w];
                }
                uint off = offset_4d(row0 + rr3, cc3, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off, r, dst_esize);
            }
        }
    }
}
