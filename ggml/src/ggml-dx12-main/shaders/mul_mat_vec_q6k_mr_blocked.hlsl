// mul_mat_vec_q6k_mr_blocked.hlsl - Block-level Q6_K matvec (M=1, 2 rows/group)
//
// Q6_K block (210 bytes): ql[128] + qh[64] + scales[16] + d(f16)
// 256 threads, 16 threads per Q6_K block, 16 elements per thread per block.
// Shares activation loads across 2 output rows.
//
// Per scale group (16 elements within a block) we compute:
//     row_acc += d * scale * (sum(q_u * x) - 32 * sum(x))
// where q_u is the 6-bit unsigned weight in [0,63] (bias of 32 deferred to
// the per-group sum to amortize one float multiply per element).
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch * ne2 * ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q6K_BSIZE   210
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

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // Thread mapping: 16 threads per Q6_K block group (one per scale group)
    uint it_size = GROUP_SIZE / 16;
    uint scale_idx = tid % 16;     // 0..15: which scale group
    uint ix = tid / 16;            // which block-group iteration

    // Per-thread element parameters (constant across the block loop)
    uint ip       = scale_idx >> 3;          // 0 or 1: which half-block
    uint sg_in_h  = scale_idx & 7u;          // 0..7: scale group within half
    uint use_high = (sg_in_h >= 4u) ? 1u : 0u;   // 0=low nibble, 1=high nibble of ql byte
    uint ql_chunk = sg_in_h & 3u;            // 0..3: which 16-byte chunk of ql in half
    uint qh_chunk = sg_in_h & 1u;            // 0 or 1: which 16-byte chunk of qh in half
    uint qh_shift = (sg_in_h & ~1u);         // 0,2,4,6 bit shift to extract high pair
    uint y_offset = scale_idx * 16;          // 16 contiguous floats per scale group

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // --- Activation: 16 contiguous floats, shared across both rows ---
        uint y_base = src1_base + (block_idx * QK_K + y_offset) * 4;
        uint4 a0 = src1.Load4(y_base);
        uint4 a1 = src1.Load4(y_base + 16);
        uint4 a2 = src1.Load4(y_base + 32);
        uint4 a3 = src1.Load4(y_base + 48);

        float by0  = asfloat(a0.x); float by1  = asfloat(a0.y);
        float by2  = asfloat(a0.z); float by3  = asfloat(a0.w);
        float by4  = asfloat(a1.x); float by5  = asfloat(a1.y);
        float by6  = asfloat(a1.z); float by7  = asfloat(a1.w);
        float by8  = asfloat(a2.x); float by9  = asfloat(a2.y);
        float by10 = asfloat(a2.z); float by11 = asfloat(a2.w);
        float by12 = asfloat(a3.x); float by13 = asfloat(a3.y);
        float by14 = asfloat(a3.z); float by15 = asfloat(a3.w);

        float xs = (by0 + by1 + by2  + by3 ) + (by4  + by5  + by6  + by7 )
                 + (by8 + by9 + by10 + by11) + (by12 + by13 + by14 + by15);

        // --- Process both rows ---
        [unroll]
        for (uint r = 0; r < NUM_ROWS; r++) {
            uint block_off = (r == 0u) ? (src0_row0 + block_idx * Q6K_BSIZE)
                                       : (src0_row1 + block_idx * Q6K_BSIZE);

            // d at offset 208 (2 bytes). Load aligned word and shift.
            uint d_off = block_off + 208;
            uint d_word = src0.Load(d_off & ~3u);
            float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

            // Scale byte: scales[scale_idx], stored at block_off + 192 + scale_idx
            uint scale_off = block_off + 192 + scale_idx;
            uint scale_word = src0.Load(scale_off & ~3u);
            uint scale_byte = (scale_word >> ((scale_off & 3u) * 8u)) & 0xFFu;
            int  scale_int = (scale_byte < 128u) ? (int)scale_byte : ((int)scale_byte - 256);
            float scale_f = float(scale_int);

            // ql: 16 contiguous bytes for this scale group
            uint ql_base = block_off + ip * 64u + ql_chunk * 16u;
            uint4 ql4 = src0.Load4(ql_base);

            // qh: 16 contiguous bytes (each holds 4 high-bit pairs for different sub-blocks)
            uint qh_base = block_off + 128u + ip * 32u + qh_chunk * 16u;
            uint4 qh4 = src0.Load4(qh_base);

            // Decode 16 elements and accumulate q_u * x
            float dot_q;
            {
                uint4 ql_lo = ql4 & 0x0F0F0F0Fu;
                uint4 ql_hi = (ql4 >> 4) & 0x0F0F0F0Fu;
                uint4 qhsel = (qh4 >> qh_shift) & 0x03030303u;

                // pick low or high nibble per element via use_high
                uint4 nibbles = (use_high == 0u) ? ql_lo : ql_hi;
                uint4 q6 = nibbles | (qhsel << 4u);

                float q0  = float( q6.x        & 0xFFu);
                float q1  = float((q6.x >> 8 ) & 0xFFu);
                float q2  = float((q6.x >> 16) & 0xFFu);
                float q3  = float( q6.x >> 24);
                float q4  = float( q6.y        & 0xFFu);
                float q5  = float((q6.y >> 8 ) & 0xFFu);
                float q6_ = float((q6.y >> 16) & 0xFFu);
                float q7  = float( q6.y >> 24);
                float q8  = float( q6.z        & 0xFFu);
                float q9  = float((q6.z >> 8 ) & 0xFFu);
                float q10 = float((q6.z >> 16) & 0xFFu);
                float q11 = float( q6.z >> 24);
                float q12 = float( q6.w        & 0xFFu);
                float q13 = float((q6.w >> 8 ) & 0xFFu);
                float q14 = float((q6.w >> 16) & 0xFFu);
                float q15 = float( q6.w >> 24);

                float s0 = mad(q0,  by0,  mad(q1,  by1,  mad(q2,  by2,  q3  * by3 )));
                float s1 = mad(q4,  by4,  mad(q5,  by5,  mad(q6_, by6,  q7  * by7 )));
                float s2 = mad(q8,  by8,  mad(q9,  by9,  mad(q10, by10, q11 * by11)));
                float s3 = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15 * by15)));
                dot_q = (s0 + s1) + (s2 + s3);
            }

            float row_acc = d * scale_f * (dot_q - 32.0f * xs);
            if (r == 0u) acc0 += row_acc;
            else         acc1 += row_acc;
        }
    }

    // Two-level reduction: wave-level + cross-wave shared memory
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
