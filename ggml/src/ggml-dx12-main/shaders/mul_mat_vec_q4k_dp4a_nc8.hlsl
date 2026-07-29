// mul_mat_vec_q4k_dp4a_nc8.hlsl - Q4_K dp4a matvec for M==8 batch.
//
// NUM_ROWS=2 x NUM_COLS=8 outputs per workgroup. Targets speculative-decoding
// draft+target workloads at npl=8.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q4K_BSIZE   144
#define Q8_1_BSIZE  36
#define NUM_ROWS    2
#define NUM_COLS    8

groupshared float shared_acc[NUM_ROWS * NUM_COLS * 64];

void decode_q4k_row(uint block_off, uint v_im, uint q_offset,
                    out float dall, out float dmin,
                    out float sc0, out float sc1, out float sc2, out float sc3,
                    out float sc4, out float sc5, out float sc6, out float sc7,
                    out uint qs0_out, out uint qs64_out) {
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

    uint qs_off = block_off + 16;
    qs0_out  = src0.Load(qs_off + q_offset);
    qs64_out = src0.Load(qs_off + q_offset + 64);
}

float compute_dp4a_row(float dall, float dmin,
                       float sc0, float sc1, float sc2, float sc3,
                       float sc4, float sc5, float sc6, float sc7,
                       uint qs0, uint qs64,
                       float q8d0, float q8d1, float q8d2, float q8d3,
                       uint q8_qs0, uint q8_qs1, uint q8_qs2, uint q8_qs3,
                       int q8_psum0, int q8_psum1, int q8_psum2, int q8_psum3) {
    uint q4_lo0  = qs0  & 0x0F0F0F0Fu;
    uint q4_hi0  = (qs0  >> 4) & 0x0F0F0F0Fu;
    uint q4_lo64 = qs64 & 0x0F0F0F0Fu;
    uint q4_hi64 = (qs64 >> 4) & 0x0F0F0F0Fu;

    int isx = 0; isx = dot4add_i8packed(q4_lo0,  q8_qs0, isx);
    int isy = 0; isy = dot4add_i8packed(q4_hi0,  q8_qs1, isy);
    int isz = 0; isz = dot4add_i8packed(q4_lo64, q8_qs2, isz);
    int isw = 0; isw = dot4add_i8packed(q4_hi64, q8_qs3, isw);

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
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

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
        float dall0, dmin0, sc00, sc01, sc02, sc03, sc04, sc05, sc06, sc07;
        uint qs0_0, qs64_0;
        decode_q4k_row(src0_row0 + block_idx * Q4K_BSIZE, v_im, q_offset,
                       dall0, dmin0, sc00, sc01, sc02, sc03, sc04, sc05, sc06, sc07,
                       qs0_0, qs64_0);

        float dall1, dmin1, sc10, sc11, sc12, sc13, sc14, sc15, sc16, sc17;
        uint qs0_1, qs64_1;
        decode_q4k_row(src0_row1 + block_idx * Q4K_BSIZE, v_im, q_offset,
                       dall1, dmin1, sc10, sc11, sc12, sc13, sc14, sc15, sc16, sc17,
                       qs0_1, qs64_1);

        [unroll] for (uint c = 0; c < NUM_COLS; c++) {
            uint q8_col_base = q8_batch_base + c * num_q8_per_vec * Q8_1_BSIZE;
            uint q8_super_base = q8_col_base + block_idx * 8 * Q8_1_BSIZE;
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

            acc[0][c] += compute_dp4a_row(dall0, dmin0, sc00, sc01, sc02, sc03, sc04, sc05, sc06, sc07,
                                          qs0_0, qs64_0, q8d0, q8d1, q8d2, q8d3,
                                          q8_qs0, q8_qs1, q8_qs2, q8_qs3,
                                          q8_psum0, q8_psum1, q8_psum2, q8_psum3);
            acc[1][c] += compute_dp4a_row(dall1, dmin1, sc10, sc11, sc12, sc13, sc14, sc15, sc16, sc17,
                                          qs0_1, qs64_1, q8d0, q8d1, q8d2, q8d3,
                                          q8_qs0, q8_qs1, q8_qs2, q8_qs3,
                                          q8_psum0, q8_psum1, q8_psum2, q8_psum3);
        }
    }

    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1) / wave_lanes;
    if (num_waves == 0) num_waves = 1;

    [unroll] for (uint rr = 0; rr < NUM_ROWS; rr++) {
        [unroll] for (uint cc = 0; cc < NUM_COLS; cc++) {
            float ws = WaveActiveSum(acc[rr][cc]);
            if (WaveIsFirstLane()) {
                shared_acc[(rr * NUM_COLS + cc) * 64 + wave_id] = ws;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        [unroll] for (uint rr = 0; rr < NUM_ROWS; rr++) {
            if (row0 + rr >= ne0) continue;
            [unroll] for (uint cc = 0; cc < NUM_COLS; cc++) {
                float r = shared_acc[(rr * NUM_COLS + cc) * 64];
                for (uint w = 1; w < num_waves; w++) {
                    r += shared_acc[(rr * NUM_COLS + cc) * 64 + w];
                }
                uint off = offset_4d(row0 + rr, cc, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off, r, dst_esize);
            }
        }
    }
}
