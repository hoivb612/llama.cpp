// mul_mat_vec_q5k_dp4a_nc.hlsli - Q5_K dp4a matvec for small-M batches.
//
// NUM_ROWS=2 x NUM_COLS outputs per workgroup. Decodes each Q5_K super-block
// once, reuses decoded weights across all activation columns. Mirrors
// mul_mat_vec_q4k_dp4a_nc.hlsli structure with Q5_K's 5th-bit qh merge.
// Wrappers set NUM_COLS.
//
// ne11 may be less than NUM_COLS - a batch is routed to the next width up, so
// n=3 runs here as NUM_COLS=4. Columns past ne11 are skipped entirely rather
// than clamped, so they cost nothing and never touch memory.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q5K_BSIZE   176
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#ifndef NUM_COLS
#define NUM_COLS    2
#endif

groupshared float shared_acc[NUM_ROWS * NUM_COLS * 64];

void decode_q5k_row(uint block_off, uint v_im, uint l0, uint q_offset,
                    out float dall, out float dmin,
                    out float sc0, out float sc1, out float sc2, out float sc3,
                    out float sc4, out float sc5, out float sc6, out float sc7,
                    out uint q5_lo,  out uint q5_hi,
                    out uint q5_lo64, out uint q5_hi64) {
    uint dm_raw = src0.Load(block_off);
    dall = f16_to_f32(dm_raw & 0xFFFFu);
    dmin = f16_to_f32(dm_raw >> 16);

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

    sc0 = float((scale_0_4_l >>  0) & 0x3Fu);
    sc1 = float((scale_0_4_l >>  8) & 0x3Fu);
    sc2 = float((scale_0_4_l >> 16) & 0x3Fu);
    sc3 = float((scale_0_4_l >> 24) & 0x3Fu);

    uint combined_8 = (((s_raw8 << 12) | s_raw8) & 0x0F0F0F0Fu) | scale_0_4_h;
    sc4 = float((combined_8 >>  0) & 0xFFu);
    sc5 = float((combined_8 >>  8) & 0xFFu);
    sc6 = float((combined_8 >> 16) & 0xFFu);
    sc7 = float((combined_8 >> 24) & 0xFFu);

    uint qs_off = block_off + 48;
    uint qs0  = src0.Load(qs_off + q_offset);
    uint qs64 = src0.Load(qs_off + q_offset + 64);

    uint qh_off = block_off + 16;
    uint qh = src0.Load(qh_off + l0);

    uint shift = 2u * v_im;
    q5_lo   = (qs0  & 0x0F0F0F0Fu)        | (((qh >> (shift + 0)) & 0x01010101u) << 4);
    q5_hi   = ((qs0  >> 4) & 0x0F0F0F0Fu) | (((qh >> (shift + 1)) & 0x01010101u) << 4);
    q5_lo64 = (qs64 & 0x0F0F0F0Fu)        | (((qh >> (shift + 4)) & 0x01010101u) << 4);
    q5_hi64 = ((qs64 >> 4) & 0x0F0F0F0Fu) | (((qh >> (shift + 5)) & 0x01010101u) << 4);
}

float compute_dp4a_row_q5k(float dall, float dmin,
                           float sc0, float sc1, float sc2, float sc3,
                           float sc4, float sc5, float sc6, float sc7,
                           uint q5_lo, uint q5_hi, uint q5_lo64, uint q5_hi64,
                           float q8d0, float q8d1, float q8d2, float q8d3,
                           uint q8_qs0, uint q8_qs1, uint q8_qs2, uint q8_qs3,
                           int q8_psum0, int q8_psum1, int q8_psum2, int q8_psum3) {
    int isx = 0; isx = dot4add_i8packed(q5_lo,   q8_qs0, isx);
    int isy = 0; isy = dot4add_i8packed(q5_hi,   q8_qs1, isy);
    int isz = 0; isz = dot4add_i8packed(q5_lo64, q8_qs2, isz);
    int isw = 0; isw = dot4add_i8packed(q5_hi64, q8_qs3, isw);

    float dot_term = mad(sc0 * q8d0, float(isx), mad(sc1 * q8d1, float(isy),
                    mad(sc4 * q8d2, float(isz), sc5 * q8d3 * float(isw))));
    float min_term = mad(sc2 * q8d0, float(q8_psum0), mad(sc3 * q8d1, float(q8_psum1),
                    mad(sc6 * q8d2, float(q8_psum2), sc7 * q8d3 * float(q8_psum3))));

    return dall * dot_term - dmin * min_term;
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
    uint num_blocks = K / QK_K;
    uint num_q8_per_vec = K / 32;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_batch_base = src1_offset +
        ((i3_q8 * ne12 + i2_q8) * ne11) * num_q8_per_vec * Q8_1_BSIZE;

    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint il = itid / 4;
    uint ir = itid % 4;
    uint v_im = il / 2;
    uint v_in = il % 2;

    uint l0 = 4 * (2 * ir + v_in);
    uint q_offset = 32 * v_im + l0;

    uint q8_sub0 = v_im * 2;
    uint q8_sub1 = q8_sub0 + 1;
    uint q8_sub2 = q8_sub0 + 4;
    uint q8_sub3 = q8_sub2 + 1;

    float acc[NUM_ROWS][NUM_COLS];
    [unroll] for (uint rr = 0; rr < NUM_ROWS; rr++) {
        [unroll] for (uint cc = 0; cc < NUM_COLS; cc++) {
            acc[rr][cc] = 0.0f;
        }
    }

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        float dall[NUM_ROWS], dmin[NUM_ROWS];
        float sc0[NUM_ROWS], sc1[NUM_ROWS], sc2[NUM_ROWS], sc3[NUM_ROWS];
        float sc4[NUM_ROWS], sc5[NUM_ROWS], sc6[NUM_ROWS], sc7[NUM_ROWS];
        uint  q5_lo[NUM_ROWS], q5_hi[NUM_ROWS], q5_lo64[NUM_ROWS], q5_hi64[NUM_ROWS];
        [unroll] for (uint r = 0; r < NUM_ROWS; r++) {
            decode_q5k_row(src0_base + (row0 + r) * nb01 + block_idx * Q5K_BSIZE,
                           v_im, l0, q_offset,
                           dall[r], dmin[r], sc0[r], sc1[r], sc2[r], sc3[r],
                           sc4[r], sc5[r], sc6[r], sc7[r],
                           q5_lo[r], q5_hi[r], q5_lo64[r], q5_hi64[r]);
        }

        [unroll] for (uint c = 0; c < NUM_COLS; c++) {
            if (c >= ne11) break;
            uint q8_super_base = q8_batch_base + c * num_q8_per_vec * Q8_1_BSIZE +
                                 block_idx * 8 * Q8_1_BSIZE;
            uint q8_off0 = q8_super_base + q8_sub0 * Q8_1_BSIZE;
            uint q8_off1 = q8_super_base + q8_sub1 * Q8_1_BSIZE;
            uint q8_off2 = q8_super_base + q8_sub2 * Q8_1_BSIZE;
            uint q8_off3 = q8_super_base + q8_sub3 * Q8_1_BSIZE;

            uint ds0 = src1.Load(q8_off0);
            uint ds1 = src1.Load(q8_off1);
            uint ds2 = src1.Load(q8_off2);
            uint ds3 = src1.Load(q8_off3);
            float q8d0 = f16_to_f32(ds0 & 0xFFFFu);
            float q8d1 = f16_to_f32(ds1 & 0xFFFFu);
            float q8d2 = f16_to_f32(ds2 & 0xFFFFu);
            float q8d3 = f16_to_f32(ds3 & 0xFFFFu);

            uint q8_qs0 = src1.Load(q8_off0 + 4 + l0);
            uint q8_qs1 = src1.Load(q8_off1 + 4 + l0);
            uint q8_qs2 = src1.Load(q8_off2 + 4 + l0);
            uint q8_qs3 = src1.Load(q8_off3 + 4 + l0);

            int q8_psum0 = 0; q8_psum0 = dot4add_i8packed(0x01010101u, q8_qs0, q8_psum0);
            int q8_psum1 = 0; q8_psum1 = dot4add_i8packed(0x01010101u, q8_qs1, q8_psum1);
            int q8_psum2 = 0; q8_psum2 = dot4add_i8packed(0x01010101u, q8_qs2, q8_psum2);
            int q8_psum3 = 0; q8_psum3 = dot4add_i8packed(0x01010101u, q8_qs3, q8_psum3);

            [unroll] for (uint r2 = 0; r2 < NUM_ROWS; r2++) {
                acc[r2][c] += compute_dp4a_row_q5k(dall[r2], dmin[r2],
                                                   sc0[r2], sc1[r2], sc2[r2], sc3[r2],
                                                   sc4[r2], sc5[r2], sc6[r2], sc7[r2],
                                                   q5_lo[r2], q5_hi[r2], q5_lo64[r2], q5_hi64[r2],
                                                   q8d0, q8d1, q8d2, q8d3,
                                                   q8_qs0, q8_qs1, q8_qs2, q8_qs3,
                                                   q8_psum0, q8_psum1, q8_psum2, q8_psum3);
            }
        }
    }

    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1) / wave_lanes;
    if (num_waves == 0) num_waves = 1;

    [unroll] for (uint rr2 = 0; rr2 < NUM_ROWS; rr2++) {
        [unroll] for (uint cc2 = 0; cc2 < NUM_COLS; cc2++) {
            if (cc2 >= ne11) break;
            float ws = WaveActiveSum(acc[rr2][cc2]);
            if (WaveIsFirstLane()) {
                shared_acc[(rr2 * NUM_COLS + cc2) * 64 + wave_id] = ws;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        [unroll] for (uint rr3 = 0; rr3 < NUM_ROWS; rr3++) {
            if (row0 + rr3 >= ne0) continue;
            [unroll] for (uint cc3 = 0; cc3 < NUM_COLS; cc3++) {
                if (cc3 >= ne11) break;
                float r = shared_acc[(rr3 * NUM_COLS + cc3) * 64];
                for (uint w = 1; w < num_waves; w++) {
                    r += shared_acc[(rr3 * NUM_COLS + cc3) * 64 + w];
                }
                uint off = offset_4d(row0 + rr3, cc3, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off, r, dst_esize);
            }
        }
    }
}