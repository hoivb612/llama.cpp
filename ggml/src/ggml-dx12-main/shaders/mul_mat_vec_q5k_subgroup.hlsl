// mul_mat_vec_q5k_subgroup.hlsl - Subgroup-cooperative Q5_K matvec (M=1)
//
// Vulkan-parity port of `mul_mat_vec_q5_k.comp` (USE_SUBGROUP_ADD_NO_SHMEM).
//
// One workgroup = one wave (32 lanes on NVIDIA RTX). Two output rows per WG
// share Q5_K activation loads. 16 lanes cooperate per Q5_K superblock; the
// other 16 lanes work on the next superblock in parallel (it_size = 2).
//
// Reduction is a single WaveActiveSum — no groupshared, no barriers.
// This is the key win over `mul_mat_vec_q5k_mr.hlsl`, which uses 256 threads
// with shared-memory tree reduction and leaves most lanes idle when
// num_blocks_per_row is small.
//
// Q5_K block (176 bytes): d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch * ne2 * ne3
// Gated on wave_size == 32 in the dispatcher (flags = 15).

#include "ggml_common.hlsli"

#define GROUP_SIZE  32
#define QK_K        256
#define Q5K_BSIZE   176

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // 16 lanes cooperate per superblock. it_size = GROUP_SIZE / 16 = 2.
    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint il = itid / 4;          // 0..3
    uint ir = itid % 4;          // 0..3
    uint v_im = il / 2;          // 0 or 1
    uint v_in = il % 2;          // 0 or 1

    // Stride-4 element layout (4-byte aligned: plain Load() is safe on all GPUs).
    // Mirrors mul_mat_vec_q5k.hlsl, NOT the stride-2 layout in the MR shader.
    uint l0 = 4 * (2 * ir + v_in);  // 0,4,8,12,16,20,24,28
    uint q_offset = 32 * v_im + l0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // ----- Load activations once, reuse across both rows -----
        // Subblocks covered: for v_im=0 → {0,1,4,5}, for v_im=1 → {2,3,6,7}.
        uint y_super = block_idx * QK_K;
        uint y_off_lo  = src1_base + (y_super + 64u * v_im      + l0) * 4u;
        uint y_off_hi  = src1_base + (y_super + 64u * v_im + 32 + l0) * 4u;
        uint y_off_lo2 = src1_base + (y_super + 64u * v_im + 128+ l0) * 4u;
        uint y_off_hi2 = src1_base + (y_super + 64u * v_im + 160+ l0) * 4u;

        uint4 y_lo  = src1.Load4(y_off_lo);
        uint4 y_hi  = src1.Load4(y_off_hi);
        uint4 y_lo2 = src1.Load4(y_off_lo2);
        uint4 y_hi2 = src1.Load4(y_off_hi2);

        float by0  = asfloat(y_lo.x);  float by1  = asfloat(y_lo.y);
        float by2  = asfloat(y_lo.z);  float by3  = asfloat(y_lo.w);
        float by4  = asfloat(y_hi.x);  float by5  = asfloat(y_hi.y);
        float by6  = asfloat(y_hi.z);  float by7  = asfloat(y_hi.w);
        float by8  = asfloat(y_lo2.x); float by9  = asfloat(y_lo2.y);
        float by10 = asfloat(y_lo2.z); float by11 = asfloat(y_lo2.w);
        float by12 = asfloat(y_hi2.x); float by13 = asfloat(y_hi2.y);
        float by14 = asfloat(y_hi2.z); float by15 = asfloat(y_hi2.w);

        // Per-subblock activation sums (used by min-correction term).
        float bsum_lo  = by0  + by1  + by2  + by3;
        float bsum_hi  = by4  + by5  + by6  + by7;
        float bsum_lo2 = by8  + by9  + by10 + by11;
        float bsum_hi2 = by12 + by13 + by14 + by15;

        // ----- Per-row weight decode + accumulate -----
        [unroll]
        for (uint r = 0; r < 2; ++r) {
            uint block_off = (r == 0 ? src0_row0 : src0_row1) + block_idx * Q5K_BSIZE;

            uint dm_raw = src0.Load(block_off);
            float dall = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

            // Scales: Q4_K-style packed decode.
            uint scales_off = block_off + 4;
            uint s_raw0, s_raw4, s_raw8;
            if (v_im == 0) {
                s_raw0 = src0.Load(scales_off)     & 0xFFFFu;
                s_raw4 = src0.Load(scales_off + 4) & 0xFFFFu;
                s_raw8 = src0.Load(scales_off + 8) & 0xFFFFu;
            } else {
                s_raw0 = (src0.Load(scales_off)     >> 16) & 0xFFFFu;
                s_raw4 = (src0.Load(scales_off + 4) >> 16) & 0xFFFFu;
                s_raw8 = (src0.Load(scales_off + 8) >> 16) & 0xFFFFu;
            }

            uint scale_0_4_l = (s_raw4 << 16) | s_raw0;
            uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0u) >> 2;

            float sc0 = float((scale_0_4_l >>  0) & 0x3Fu);
            float sc1 = float((scale_0_4_l >>  8) & 0x3Fu);
            float sc2 = float((scale_0_4_l >> 16) & 0x3Fu);
            float sc3 = float((scale_0_4_l >> 24) & 0x3Fu);

            uint combined_8 = (((s_raw8 << 12) | s_raw8) & 0x0F0F0F0Fu) | scale_0_4_h;
            float sc4 = float((combined_8 >>  0) & 0xFFu);
            float sc5 = float((combined_8 >>  8) & 0xFFu);
            float sc6 = float((combined_8 >> 16) & 0xFFu);
            float sc7 = float((combined_8 >> 24) & 0xFFu);

            // qs (4-bit nibbles) at block_off + 48; qh (high bits) at block_off + 16.
            uint qs_off = block_off + 48;
            uint qs0  = src0.Load(qs_off + q_offset);
            uint qs64 = src0.Load(qs_off + q_offset + 64);

            uint qh_off = block_off + 16;
            uint qh = src0.Load(qh_off + l0);

            uint shift = 2u * v_im;
            uint q5_lo   = (qs0  & 0x0F0F0F0Fu)        | (((qh >> (shift + 0)) & 0x01010101u) << 4);
            uint q5_hi   = ((qs0  >> 4) & 0x0F0F0F0Fu) | (((qh >> (shift + 1)) & 0x01010101u) << 4);
            uint q5_lo64 = (qs64 & 0x0F0F0F0Fu)        | (((qh >> (shift + 4)) & 0x01010101u) << 4);
            uint q5_hi64 = ((qs64 >> 4) & 0x0F0F0F0Fu) | (((qh >> (shift + 5)) & 0x01010101u) << 4);

            float q0  = float((q5_lo   >>  0) & 0xFFu);
            float q1  = float((q5_lo   >>  8) & 0xFFu);
            float q2  = float((q5_lo   >> 16) & 0xFFu);
            float q3  = float((q5_lo   >> 24) & 0xFFu);
            float q4  = float((q5_hi   >>  0) & 0xFFu);
            float q5  = float((q5_hi   >>  8) & 0xFFu);
            float q6  = float((q5_hi   >> 16) & 0xFFu);
            float q7  = float((q5_hi   >> 24) & 0xFFu);
            float q8  = float((q5_lo64 >>  0) & 0xFFu);
            float q9  = float((q5_lo64 >>  8) & 0xFFu);
            float q10 = float((q5_lo64 >> 16) & 0xFFu);
            float q11 = float((q5_lo64 >> 24) & 0xFFu);
            float q12 = float((q5_hi64 >>  0) & 0xFFu);
            float q13 = float((q5_hi64 >>  8) & 0xFFu);
            float q14 = float((q5_hi64 >> 16) & 0xFFu);
            float q15 = float((q5_hi64 >> 24) & 0xFFu);

            float sx = q0*by0 + q1*by1 + q2*by2 + q3*by3;
            float sy = q4*by4 + q5*by5 + q6*by6 + q7*by7;
            float sz = q8*by8 + q9*by9 + q10*by10 + q11*by11;
            float sw = q12*by12 + q13*by13 + q14*by14 + q15*by15;

            float smin = sc2 * bsum_lo + sc3 * bsum_hi + sc6 * bsum_lo2 + sc7 * bsum_hi2;

            float row_acc = dall * (sx*sc0 + sy*sc1 + sz*sc4 + sw*sc5) - dmin * smin;
            if (r == 0) acc0 += row_acc;
            else        acc1 += row_acc;
        }
    }

    // Single-wave reduction: GROUP_SIZE == WARP_SIZE on the dispatched-to GPU
    // (gated by wave_size==32 in dispatcher). No shared memory, no barrier.
    float result0 = WaveActiveSum(acc0);
    float result1 = WaveActiveSum(acc1);

    if (tid == 0) {
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
