// mul_mat_q2k.hlsl - Cooperative batch matmul for Q2_K quantized weights
//
// 16 output rows per group, 16 K-threads cooperate per row.
// Processes TILE_M=4 activation columns per group — weight data is decoded once
// per K-block and reused across all 4 columns, saving ~36% instructions.
// Uses Load2 for f32 activations (8 paired loads instead of 16 individual).
//
// Q2_K block layout (84 bytes per 256 elements):
//   scales[16]: 4-bit scale (low) + 4-bit min (high)
//   qs[64]:     2-bit quantized values, 4 per byte
//   d (f16) + dmin (f16): super-block scale and min
//
// Dispatch: groups_x = ceil(N/16), groups_y = ceil(M/4), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE       256
#define ROWS_PER_GROUP   16
#define THREADS_PER_ROW  16
#define QK_K             256
#define Q2K_BSIZE        84
#define TILE_M           4

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row_local = tid / THREADS_PER_ROW;
    uint k_tid     = tid % THREADS_PER_ROW;

    uint i0      = group_id.x * ROWS_PER_GROUP + row_local;
    uint i1_base = group_id.y * TILE_M;
    uint i2      = group_id.z % ne2;
    uint i3      = group_id.z / ne2;

    uint N = ne0;
    uint M = ne1;
    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_bc  = i2 * nb12 + i3 * nb13;

    // Thread mapping (same as matvec)
    uint v_im     = k_tid / 8;
    uint v_in     = k_tid % 8;
    uint l0       = 2 * v_in;
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 128 * v_im + l0;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    if (i0 < N) {
        for (uint bg = 0; bg < num_blocks; bg++) {
            uint block_off = src0_row + bg * Q2K_BSIZE;

            // --- Weight decode (once, reused for all TILE_M columns) ---
            uint dm_raw = src0.Load(block_off + 80);
            float d    = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

            uint sc_base  = block_off + 8 * v_im;
            uint sc_word0 = src0.Load(sc_base);
            uint sc_word1 = src0.Load(sc_base + 4);

            float sc0 = float((sc_word0 >>  0) & 0xFu);
            float sc1 = float((sc_word0 >>  8) & 0xFu);
            float sc2 = float((sc_word0 >> 16) & 0xFu);
            float sc3 = float((sc_word0 >> 24) & 0xFu);
            float sc4 = float((sc_word1 >>  0) & 0xFu);
            float sc5 = float((sc_word1 >>  8) & 0xFu);
            float sc6 = float((sc_word1 >> 16) & 0xFu);
            float sc7 = float((sc_word1 >> 24) & 0xFu);

            float mn0 = float((sc_word0 >>  4) & 0xFu);
            float mn1 = float((sc_word0 >> 12) & 0xFu);
            float mn2 = float((sc_word0 >> 20) & 0xFu);
            float mn3 = float((sc_word0 >> 28) & 0xFu);
            float mn4 = float((sc_word1 >>  4) & 0xFu);
            float mn5 = float((sc_word1 >> 12) & 0xFu);
            float mn6 = float((sc_word1 >> 20) & 0xFu);
            float mn7 = float((sc_word1 >> 28) & 0xFu);

            uint qs_abs = block_off + 16;
            uint addr_a = qs_abs + q_offset;
            uint word_a = src0.Load(addr_a & ~3u);
            uint pair_a = (word_a >> ((addr_a & 2u) * 8u)) & 0xFFFFu;
            uint addr_b = qs_abs + q_offset + 16;
            uint word_b = src0.Load(addr_b & ~3u);
            uint pair_b = (word_b >> ((addr_b & 2u) * 8u)) & 0xFFFFu;
            uint qs_u32 = pair_a | (pair_b << 16);

            float q_0  = float( qs_u32        & 3u);
            float q_1  = float((qs_u32 >>  8) & 3u);
            float q_2  = float((qs_u32 >> 16) & 3u);
            float q_3  = float((qs_u32 >> 24) & 3u);
            float q_4  = float((qs_u32 >>  2) & 3u);
            float q_5  = float((qs_u32 >> 10) & 3u);
            float q_6  = float((qs_u32 >> 18) & 3u);
            float q_7  = float((qs_u32 >> 26) & 3u);
            float q_8  = float((qs_u32 >>  4) & 3u);
            float q_9  = float((qs_u32 >> 12) & 3u);
            float q_10 = float((qs_u32 >> 20) & 3u);
            float q_11 = float((qs_u32 >> 28) & 3u);
            float q_12 = float((qs_u32 >>  6) & 3u);
            float q_13 = float((qs_u32 >> 14) & 3u);
            float q_14 = float((qs_u32 >> 22) & 3u);
            float q_15 = float((qs_u32 >> 30) & 3u);

            // --- Process TILE_M activation columns ---
            uint y_elem = bg * QK_K + y_offset;

            [unroll]
            for (uint m = 0; m < TILE_M; m++) {
                if (i1_base + m >= M) break;

                uint src1_m = src1_offset + (i1_base + m) * nb11 + src1_bc;
                float by0, by1, by2, by3, by4, by5, by6, by7;
                float by8, by9, by10, by11, by12, by13, by14, by15;

                if (src1_esize == 4u) {
                    uint y_off = src1_m + y_elem * 4u;
                    uint2 a0 = src1.Load2(y_off);
                    uint2 a1 = src1.Load2(y_off + 64);
                    uint2 a2 = src1.Load2(y_off + 128);
                    uint2 a3 = src1.Load2(y_off + 192);
                    uint2 a4 = src1.Load2(y_off + 256);
                    uint2 a5 = src1.Load2(y_off + 320);
                    uint2 a6 = src1.Load2(y_off + 384);
                    uint2 a7 = src1.Load2(y_off + 448);
                    by0 =asfloat(a0.x); by1 =asfloat(a0.y);
                    by2 =asfloat(a1.x); by3 =asfloat(a1.y);
                    by4 =asfloat(a2.x); by5 =asfloat(a2.y);
                    by6 =asfloat(a3.x); by7 =asfloat(a3.y);
                    by8 =asfloat(a4.x); by9 =asfloat(a4.y);
                    by10=asfloat(a5.x); by11=asfloat(a5.y);
                    by12=asfloat(a6.x); by13=asfloat(a6.y);
                    by14=asfloat(a7.x); by15=asfloat(a7.y);
                } else {
                    uint yb = src1_m + y_elem * src1_esize;
                    by0  = load_auto(src1, yb,                    src1_esize);
                    by1  = load_auto(src1, yb +      src1_esize,  src1_esize);
                    by2  = load_auto(src1, yb + 16 * src1_esize,  src1_esize);
                    by3  = load_auto(src1, yb + 17 * src1_esize,  src1_esize);
                    by4  = load_auto(src1, yb + 32 * src1_esize,  src1_esize);
                    by5  = load_auto(src1, yb + 33 * src1_esize,  src1_esize);
                    by6  = load_auto(src1, yb + 48 * src1_esize,  src1_esize);
                    by7  = load_auto(src1, yb + 49 * src1_esize,  src1_esize);
                    by8  = load_auto(src1, yb + 64 * src1_esize,  src1_esize);
                    by9  = load_auto(src1, yb + 65 * src1_esize,  src1_esize);
                    by10 = load_auto(src1, yb + 80 * src1_esize,  src1_esize);
                    by11 = load_auto(src1, yb + 81 * src1_esize,  src1_esize);
                    by12 = load_auto(src1, yb + 96 * src1_esize,  src1_esize);
                    by13 = load_auto(src1, yb + 97 * src1_esize,  src1_esize);
                    by14 = load_auto(src1, yb + 112 * src1_esize, src1_esize);
                    by15 = load_auto(src1, yb + 113 * src1_esize, src1_esize);
                }

                float s1 = mad(sc0, mad(q_0, by0, q_1 * by1),
                           mad(sc1, mad(q_2, by2, q_3 * by3),
                           mad(sc2, mad(q_4, by4, q_5 * by5),
                           mad(sc3, mad(q_6, by6, q_7 * by7),
                           mad(sc4, mad(q_8, by8, q_9 * by9),
                           mad(sc5, mad(q_10, by10, q_11 * by11),
                           mad(sc6, mad(q_12, by12, q_13 * by13),
                               sc7 * mad(q_14, by14, q_15 * by15))))))));

                float s2 = mad(mn0, by0 + by1,
                           mad(mn1, by2 + by3,
                           mad(mn2, by4 + by5,
                           mad(mn3, by6 + by7,
                           mad(mn4, by8 + by9,
                           mad(mn5, by10 + by11,
                           mad(mn6, by12 + by13,
                               mn7 * (by14 + by15))))))));

                float partial = d * s1 - dmin * s2;
                if (m == 0u) acc0 += partial;
                else if (m == 1u) acc1 += partial;
                else if (m == 2u) acc2 += partial;
                else acc3 += partial;
            }
        }
    }

    // Butterfly reduction across 16 K-threads for all 4 accumulators
    uint wave_lane = WaveGetLaneIndex();
    [unroll]
    for (uint offset = 1; offset < THREADS_PER_ROW; offset <<= 1) {
        acc0 += WaveReadLaneAt(acc0, wave_lane ^ offset);
        acc1 += WaveReadLaneAt(acc1, wave_lane ^ offset);
        acc2 += WaveReadLaneAt(acc2, wave_lane ^ offset);
        acc3 += WaveReadLaneAt(acc3, wave_lane ^ offset);
    }

    if (k_tid == 0 && i0 < N) {
        if (i1_base     < M) store_auto(dst, offset_4d(i0, i1_base,   i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc0, dst_esize);
        if (i1_base + 1 < M) store_auto(dst, offset_4d(i0, i1_base+1, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc1, dst_esize);
        if (i1_base + 2 < M) store_auto(dst, offset_4d(i0, i1_base+2, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc2, dst_esize);
        if (i1_base + 3 < M) store_auto(dst, offset_4d(i0, i1_base+3, i2, i3, nb0, nb1, nb2, nb3, dst_offset), acc3, dst_esize);
    }
}
