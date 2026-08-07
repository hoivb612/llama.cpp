// mul_mat_vec_q4k_mr.hlsl - Multi-row matrix-vector multiply for Q4_K weights (M=1)
//
// Processes 2 output rows per workgroup, sharing activation loads.
// Uses Load4 for vectorized activation reads and mad() for FMA.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3
//
// RMS_FUSED variant (mul_mat_vec_q4k_mr_rms.hlsl, fl=102): the preceding
// RMS_NORM+MUL dispatch is absorbed. src1 then carries the *un-normalized*
// activation x, src6 the norm weight g and op14 the epsilon. Because the RMS
// scale is a scalar over the row, dot(w, (x/rms)*g) == (1/rms) * sum_k(w_k *
// g_k * x_k), so the existing pass accumulates sum(x*x) alongside the dot
// product and applies the scale once at the end. The 16 threads of a block
// partition its 256 elements exactly, and the ix groups partition the blocks,
// so the group-wide sum covers every k exactly once.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q4K_BSIZE   144
#define NUM_ROWS    2

#if RMS_FUSED
groupshared float shared_acc[96];  // 3 * max_waves (rows 0/1 + sum(x*x))
#else
groupshared float shared_acc[64];  // 2 * max_waves (max 32 waves for wave_size=8)
#endif

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
#if RMS_FUSED
    float ss = 0.0f;
#endif

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

#if RMS_FUSED
        ss = mad(by0, by0, mad(by1, by1, mad(by2, by2, mad(by3, by3,
             mad(by4, by4, mad(by5, by5, mad(by6, by6, mad(by7, by7,
             mad(by8, by8, mad(by9, by9, mad(by10, by10, mad(by11, by11,
             mad(by12, by12, mad(by13, by13, mad(by14, by14,
             mad(by15, by15, ss))))))))))))))));

        uint g1_off = (block_idx * QK_K + y_offset) * 4;
        uint g2_off = g1_off + 128 * 4;
        float4 g0 = asfloat(src6.Load4(g1_off));
        float4 g1 = asfloat(src6.Load4(g1_off + 128));
        float4 g2 = asfloat(src6.Load4(g2_off));
        float4 g3 = asfloat(src6.Load4(g2_off + 128));

        by0  *= g0.x; by1  *= g0.y; by2  *= g0.z; by3  *= g0.w;
        by4  *= g1.x; by5  *= g1.y; by6  *= g1.z; by7  *= g1.w;
        by8  *= g2.x; by9  *= g2.y; by10 *= g2.z; by11 *= g2.w;
        by12 *= g3.x; by13 *= g3.y; by14 *= g3.z; by15 *= g3.w;
#endif

        // --- Process both rows ---
        [unroll]
        for (uint r = 0; r < NUM_ROWS; r++) {
            uint block_off = (r == 0 ? src0_row0 : src0_row1) + block_idx * Q4K_BSIZE;

            // Block header is d(2) + dmin(2) + scales(12) = 16 bytes, so a
            // single Load4 covers what was 4 separate 32-bit loads.
            uint4 hdr = src0.Load4(block_off);
            float dall = f16_to_f32(hdr.x & 0xFFFFu);
            float dmin = f16_to_f32(hdr.x >> 16);

            // Decode scales (12 bytes at block_off + 4)
            uint s_raw0, s_raw4, s_raw8;
            if (v_im == 0) {
                s_raw0 = hdr.y & 0xFFFFu;
                s_raw4 = hdr.z & 0xFFFFu;
                s_raw8 = hdr.w & 0xFFFFu;
            } else {
                s_raw0 = (hdr.y >> 16) & 0xFFFFu;
                s_raw4 = (hdr.z >> 16) & 0xFFFFu;
                s_raw8 = (hdr.w >> 16) & 0xFFFFu;
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
#if RMS_FUSED
    float wave_ss = WaveActiveSum(ss);
#endif
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
#if RMS_FUSED
        shared_acc[64 + wave_id] = wave_ss;
#endif
    }
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction on shared memory
    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
#if RMS_FUSED
            shared_acc[64 + tid] += shared_acc[64 + tid + s];
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
#if RMS_FUSED
        float rms_scale = 1.0f / sqrt(shared_acc[64] / (float)K + asfloat(op14));
#endif
        // Row 0
        float result0 = shared_acc[0];
#if RMS_FUSED
        result0 *= rms_scale;
#endif
        result0 += load_fused_bias(row0, i2, i3);

        // Row 1 (guard for odd N)
        bool has_row1 = (row0 + 1 < ne0);
        float result1 = 0.0f;
        if (has_row1) {
            result1 = shared_acc[32];
#if RMS_FUSED
            result1 *= rms_scale;
#endif
            result1 += load_fused_bias(row0 + 1, i2, i3);
        }

        if (mmv_scatter_active()) {
            mmv_store_scatter(row0, 0u, result0);
            if (has_row1) {
                mmv_store_scatter(row0 + 1, 0u, result1);
            }
        } else {
#if QK_SPLIT
            // Rows [0, op2) are the Q projection and land at dst_offset; the
            // rest are the K projection, a separate tensor in the same buffer
            // based at op3. Region boundaries are head-dim aligned, so a row
            // pair never straddles them, but each row picks its own base to
            // stay correct for odd splits.
            uint q_rows = op2;
            uint base0  = (row0 < q_rows) ? dst_offset : op3;
            uint local0 = (row0 < q_rows) ? row0 : (row0 - q_rows);
            store_auto(dst, base0 + local0 * nb0, result0, dst_esize);
            if (has_row1) {
                uint r1     = row0 + 1u;
                uint base1  = (r1 < q_rows) ? dst_offset : op3;
                uint local1 = (r1 < q_rows) ? r1 : (r1 - q_rows);
                store_auto(dst, base1 + local1 * nb0, result1, dst_esize);
            }
#else
            uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d0, result0, dst_esize);
            if (has_row1) {
                uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, off_d1, result1, dst_esize);
            }
#endif
        }
    }
}
