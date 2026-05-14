// mul_mat_vec_q4k_mr.hlsl - Multi-row matrix-vector multiply for Q4_K weights (M=1)
//
// Processes 2 output rows per workgroup, sharing activation loads.
// Uses Load4 for vectorized activation reads and mad() for FMA.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q4K_BSIZE   144
#define NUM_ROWS    2

groupshared float shared_acc[64];  // 2 * max_waves (max 32 waves for wave_size=8)

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

    // Weight row bases
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    // Activation base (shared between rows)
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // Thread mapping: 16 threads per Q4_K block
    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    uint il = itid / 4;
    uint ir = itid % 4;
    uint v_im = il / 2;
    uint v_in = il % 2;

    uint l0 = 4 * (2 * ir + v_in);
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 64 * v_im + l0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // --- Load activation data once (shared across both rows) ---
        uint y1_off = src1_base + (block_idx * QK_K + y_offset) * 4;
        uint y2_off = y1_off + 128 * 4;

        // Load4: 4 consecutive F32 values per call (vs Load2 in original)
        uint4 a0 = src1.Load4(y1_off);
        uint4 a1 = src1.Load4(y1_off + 128);
        uint4 a2 = src1.Load4(y2_off);
        uint4 a3 = src1.Load4(y2_off + 128);

        float by0  = asfloat(a0.x); float by1  = asfloat(a0.y);
        float by2  = asfloat(a0.z); float by3  = asfloat(a0.w);
        float by4  = asfloat(a1.x); float by5  = asfloat(a1.y);
        float by6  = asfloat(a1.z); float by7  = asfloat(a1.w);
        float by8  = asfloat(a2.x); float by9  = asfloat(a2.y);
        float by10 = asfloat(a2.z); float by11 = asfloat(a2.w);
        float by12 = asfloat(a3.x); float by13 = asfloat(a3.y);
        float by14 = asfloat(a3.z); float by15 = asfloat(a3.w);

        // --- Process both rows ---
        [unroll]
        for (uint r = 0; r < NUM_ROWS; r++) {
            uint block_off = (r == 0 ? src0_row0 : src0_row1) + block_idx * Q4K_BSIZE;

            // Load block header (d, dmin as packed f16 pair)
            uint dm_raw = src0.Load(block_off);
            float dall = f16_to_f32(dm_raw & 0xFFFFu);
            float dmin = f16_to_f32(dm_raw >> 16);

            // Decode scales (12 bytes at block_off + 4)
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

            float sc0 = float((scale_0_4_l >>  0) & 0x3Fu);
            float sc1 = float((scale_0_4_l >>  8) & 0x3Fu);
            float sc2 = float((scale_0_4_l >> 16) & 0x3Fu);
            float sc3 = float((scale_0_4_l >> 24) & 0x3Fu);

            uint combined_8 = (((s_raw8 << 12) | s_raw8) & 0x0F0F0F0Fu) | scale_0_4_h;
            float sc4 = float((combined_8 >>  0) & 0xFFu);
            float sc5 = float((combined_8 >>  8) & 0xFFu);
            float sc6 = float((combined_8 >> 16) & 0xFFu);
            float sc7 = float((combined_8 >> 24) & 0xFFu);

            // Load qs (2 uint32s = 16 nibbles)
            uint qs_off = block_off + 16;
            uint qs0  = src0.Load(qs_off + q_offset);
            uint qs64 = src0.Load(qs_off + q_offset + 64);

            // Extract 16 4-bit values using masked unpacking
            uint qs0_lo = qs0 & 0x0F0F0F0Fu;
            uint qs0_hi = (qs0 >> 4) & 0x0F0F0F0Fu;
            uint qs64_lo = qs64 & 0x0F0F0F0Fu;
            uint qs64_hi = (qs64 >> 4) & 0x0F0F0F0Fu;

            float q0  = float(qs0_lo & 0xFFu);
            float q1  = float((qs0_lo >> 8) & 0xFFu);
            float q2  = float((qs0_lo >> 16) & 0xFFu);
            float q3  = float(qs0_lo >> 24);
            float q4  = float(qs0_hi & 0xFFu);
            float q5  = float((qs0_hi >> 8) & 0xFFu);
            float q6  = float((qs0_hi >> 16) & 0xFFu);
            float q7  = float(qs0_hi >> 24);
            float q8  = float(qs64_lo & 0xFFu);
            float q9  = float((qs64_lo >> 8) & 0xFFu);
            float q10 = float((qs64_lo >> 16) & 0xFFu);
            float q11 = float(qs64_lo >> 24);
            float q12 = float(qs64_hi & 0xFFu);
            float q13 = float((qs64_hi >> 8) & 0xFFu);
            float q14 = float((qs64_hi >> 16) & 0xFFu);
            float q15 = float(qs64_hi >> 24);

            // Dot products using mad()
            float sx = mad(q0, by0, mad(q1, by1, mad(q2, by2, q3 * by3)));
            float sy = mad(q4, by4, mad(q5, by5, mad(q6, by6, q7 * by7)));
            float sz = mad(q8, by8, mad(q9, by9, mad(q10, by10, q11 * by11)));
            float sw = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15 * by15)));

            // Min compensation
            float smin = mad(sc2, by0+by1+by2+by3, mad(sc3, by4+by5+by6+by7,
                        mad(sc6, by8+by9+by10+by11, sc7 * (by12+by13+by14+by15))));

            float row_acc = mad(dall, mad(sx, sc0, mad(sy, sc1, mad(sz, sc4, sw * sc5))), -dmin * smin);
            if (r == 0) acc0 += row_acc;
            else        acc1 += row_acc;
        }
    }

    // Two-level reduction with tree reduction for cross-vendor correctness
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction on shared memory
    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        // Row 0
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        // Row 1 (guard for odd N)
        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
