// mul_mat_vec_q5k_dp4a.hlsl - dp4a-accelerated Q5_K matvec (M=1)
//
// Uses dot4add_i8packed (SM 6.4) for integer dot products.
// Same structure as Q4_K dp4a; the 5th bit lives in qh[] and is merged
// into the low/high 4-bit nibbles to form a 5-bit unsigned value [0..31]
// that fits in signed int8 (which dot4add_i8packed expects).
//
// Processes 2 output rows per workgroup, sharing Q8_1 activation loads.
// src1 is pre-quantized Q8_1 data in a scratch buffer.
//
// Q5_K block (176 bytes): d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
// Q8_1 block (36 bytes):  ds(2xf16 packed) + qs[32](int8 packed as 8 x uint32)
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q5K_BSIZE   176
#define Q8_1_BSIZE  36
#define NUM_ROWS    2

// Wave-portable reduction LDS. See mul_mat_vec_q4k_dp4a.hlsl for rationale.
groupshared float shared_acc0[64];
groupshared float shared_acc1[64];

// Decode Q5_K scales + 5-bit values for one row's superblock.
// q_offset and l0 use stride-4 indexing (l0 = 4*(2*ir+v_in)) so all
// addresses are 4-byte aligned — plain Load() is safe.
//
// Outputs:
//   dall, dmin           : block scales
//   sc0..sc7             : per-subblock scales
//   q5_lo, q5_hi         : packed 5-bit values for subblocks (sub_lo, sub_hi)
//                          covering the first half of the superblock
//   q5_lo64, q5_hi64     : same for the second half (sub+4)
void decode_q5k_row(uint block_off, uint v_im, uint l0, uint q_offset,
                    out float dall, out float dmin,
                    out float sc0, out float sc1, out float sc2, out float sc3,
                    out float sc4, out float sc5, out float sc6, out float sc7,
                    out uint q5_lo,  out uint q5_hi,
                    out uint q5_lo64, out uint q5_hi64) {
    uint dm_raw = src0.Load(block_off);
    dall = f16_to_f32(dm_raw & 0xFFFFu);
    dmin = f16_to_f32(dm_raw >> 16);

    // Scales: identical to Q4_K layout
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

    // qs[128] starts at block_off + 16 (header) + 32 (qh) = block_off + 48
    uint qs_off = block_off + 48;
    uint qs0  = src0.Load(qs_off + q_offset);
    uint qs64 = src0.Load(qs_off + q_offset + 64);

    // qh[32] starts at block_off + 16 (after header). Each byte covers
    // 8 elements (1 bit each); l0 in {0,4,...,28} addresses 4 bytes.
    uint qh_off = block_off + 16;
    uint qh = src0.Load(qh_off + l0);

    // Merge 5th bit into the 4-bit nibble for each of the 4 subblocks
    // we touch. shift = 2*v_im picks the low/high nibble pair within qh.
    uint shift = 2u * v_im;
    q5_lo   = (qs0  & 0x0F0F0F0Fu)        | (((qh >> (shift + 0)) & 0x01010101u) << 4);
    q5_hi   = ((qs0  >> 4) & 0x0F0F0F0Fu) | (((qh >> (shift + 1)) & 0x01010101u) << 4);
    q5_lo64 = (qs64 & 0x0F0F0F0Fu)        | (((qh >> (shift + 4)) & 0x01010101u) << 4);
    q5_hi64 = ((qs64 >> 4) & 0x0F0F0F0Fu) | (((qh >> (shift + 5)) & 0x01010101u) << 4);
}

// Compute dp4a accumulation for one row (mirrors compute_dp4a_row in Q4_K)
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
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_q8_per_vec * Q8_1_BSIZE;

    // 16 threads cooperate per Q5_K superblock (256 elems / 16 = 16 elems/thread)
    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint il = itid / 4;
    uint ir = itid % 4;
    uint v_im = il / 2;
    uint v_in = il % 2;

    // Stride-4 element layout (matches Q4_K dp4a, NOT the stride-2 used by
    // the float Q5_K shader). 4-byte aligned addresses → plain Load() OK.
    uint l0 = 4 * (2 * ir + v_in);  // 0,4,8,...,28
    uint q_offset = 32 * v_im + l0;

    uint q8_sub0 = v_im * 2;
    uint q8_sub1 = q8_sub0 + 1;
    uint q8_sub2 = q8_sub0 + 4;
    uint q8_sub3 = q8_sub2 + 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // --- Load Q8_1 data once (shared across both rows) ---
        uint q8_super_base = q8_vec_base + block_idx * 8 * Q8_1_BSIZE;

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

        // psum_i = sum of 4 Q8 bytes (used for the dmin correction term)
        int q8_psum0 = 0; q8_psum0 = dot4add_i8packed(0x01010101u, q8_qs0, q8_psum0);
        int q8_psum1 = 0; q8_psum1 = dot4add_i8packed(0x01010101u, q8_qs1, q8_psum1);
        int q8_psum2 = 0; q8_psum2 = dot4add_i8packed(0x01010101u, q8_qs2, q8_psum2);
        int q8_psum3 = 0; q8_psum3 = dot4add_i8packed(0x01010101u, q8_qs3, q8_psum3);

        // --- Row 0 ---
        {
            float dall, dmin, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7;
            uint q5_lo, q5_hi, q5_lo64, q5_hi64;
            decode_q5k_row(src0_row0 + block_idx * Q5K_BSIZE, v_im, l0, q_offset,
                           dall, dmin, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7,
                           q5_lo, q5_hi, q5_lo64, q5_hi64);
            acc0 += compute_dp4a_row_q5k(dall, dmin, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7,
                                         q5_lo, q5_hi, q5_lo64, q5_hi64,
                                         q8d0, q8d1, q8d2, q8d3,
                                         q8_qs0, q8_qs1, q8_qs2, q8_qs3,
                                         q8_psum0, q8_psum1, q8_psum2, q8_psum3);
        }

        // --- Row 1 ---
        {
            float dall, dmin, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7;
            uint q5_lo, q5_hi, q5_lo64, q5_hi64;
            decode_q5k_row(src0_row1 + block_idx * Q5K_BSIZE, v_im, l0, q_offset,
                           dall, dmin, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7,
                           q5_lo, q5_hi, q5_lo64, q5_hi64);
            acc1 += compute_dp4a_row_q5k(dall, dmin, sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7,
                                         q5_lo, q5_hi, q5_lo64, q5_hi64,
                                         q8d0, q8d1, q8d2, q8d3,
                                         q8_qs0, q8_qs1, q8_qs2, q8_qs3,
                                         q8_psum0, q8_psum1, q8_psum2, q8_psum3);
        }
    }

    // Wave-portable reduction. See mul_mat_vec_q4k_dp4a.hlsl for rationale.
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_lanes = WaveGetLaneCount();
    uint wave_id = tid / wave_lanes;
    uint num_waves = (GROUP_SIZE + wave_lanes - 1) / wave_lanes;
    if (num_waves == 0) num_waves = 1;

    if (WaveIsFirstLane()) {
        shared_acc0[wave_id] = wave_sum0;
        shared_acc1[wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result0 = shared_acc0[0];
        for (uint w = 1; w < num_waves; w++) result0 += shared_acc0[w];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc1[0];
            for (uint w = 1; w < num_waves; w++) result1 += shared_acc1[w];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
