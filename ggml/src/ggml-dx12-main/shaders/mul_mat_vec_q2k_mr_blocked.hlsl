// mul_mat_vec_q2k_mr_blocked.hlsl - Block-level Q2_K matvec (M=1, 2 rows/group)
//
// Q2_K block (84 bytes per 256 elements, 4-byte aligned):
//   offset  0..15  : scales[16] (low nibble = scale, high nibble = ml)
//   offset 16..79  : qs[64]     (2 bits per element)
//   offset 80..81  : d    (fp16)
//   offset 82..83  : dmin (fp16)
//
// 256 threads, 16 threads per Q2_K block, 16 elements per thread.
// Mirrors the Q4_K/Q5_K/Q6_K MR layout: each lane decodes one scale group
// (16 contiguous elements) with a vectorised qs Load4, sharing one activation
// Load4 stream across two output rows. Per-scale-group bias factorisation:
//     row_acc += scale * sum(q * x) - ml * sum(x)
// avoids per-element ml subtractions.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch * ne2 * ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q2K_BSIZE   84
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

    // Thread mapping: 16 threads per Q2_K block (one per scale group)
    uint it_size = GROUP_SIZE / 16;
    uint scale_idx = tid % 16;
    uint ix = tid / 16;

    // Per-thread element parameters (constant across the block loop)
    uint n      = scale_idx >> 3;
    uint j_h    = scale_idx & 7u;
    uint j      = j_h >> 1;
    uint half_  = j_h & 1u;
    uint shift  = j << 1;
    uint qs_chunk_off = 16u + (n << 5) + (half_ << 4); // 16 contiguous qs bytes
    uint y_offset = scale_idx * 16;                    // 16 contiguous floats

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
            uint block_off = (r == 0u) ? (src0_row0 + block_idx * Q2K_BSIZE)
                                       : (src0_row1 + block_idx * Q2K_BSIZE);

            // d, dmin packed into 4-byte word at offset 80
            uint dm = src0.Load(block_off + 80u);
            float d    = f16_to_f32(dm & 0xFFFFu);
            float dmin = f16_to_f32(dm >> 16);

            // scales[scale_idx]: 1 byte at offset block_off + scale_idx
            uint scales_off = block_off + scale_idx;
            uint scales_w   = src0.Load(scales_off & ~3u);
            uint sc_byte    = (scales_w >> ((scales_off & 3u) * 8u)) & 0xFFu;
            float scale = d    * float(sc_byte & 0x0Fu);
            float ml    = dmin * float(sc_byte >> 4);

            // qs: 16 contiguous bytes for this scale group (4-byte aligned)
            uint4 qs4 = src0.Load4(block_off + qs_chunk_off);

            // Extract per-element 2-bit qs values at shift = j*2
            uint4 q4 = (qs4 >> shift) & 0x03030303u;

            float dot_q;
            {
                float q0  = float( q4.x        & 0xFFu);
                float q1  = float((q4.x >> 8 ) & 0xFFu);
                float q2  = float((q4.x >> 16) & 0xFFu);
                float q3  = float( q4.x >> 24);
                float q4_ = float( q4.y        & 0xFFu);
                float q5  = float((q4.y >> 8 ) & 0xFFu);
                float q6  = float((q4.y >> 16) & 0xFFu);
                float q7  = float( q4.y >> 24);
                float q8  = float( q4.z        & 0xFFu);
                float q9  = float((q4.z >> 8 ) & 0xFFu);
                float q10 = float((q4.z >> 16) & 0xFFu);
                float q11 = float( q4.z >> 24);
                float q12 = float( q4.w        & 0xFFu);
                float q13 = float((q4.w >> 8 ) & 0xFFu);
                float q14 = float((q4.w >> 16) & 0xFFu);
                float q15 = float( q4.w >> 24);

                float s0 = mad(q0,  by0,  mad(q1,  by1,  mad(q2,  by2,  q3  * by3 )));
                float s1 = mad(q4_, by4,  mad(q5,  by5,  mad(q6,  by6,  q7  * by7 )));
                float s2 = mad(q8,  by8,  mad(q9,  by9,  mad(q10, by10, q11 * by11)));
                float s3 = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15 * by15)));
                dot_q = (s0 + s1) + (s2 + s3);
            }

            float row_contrib = scale * dot_q - ml * xs;
            if (r == 0u) acc0 += row_contrib;
            else         acc1 += row_contrib;
        }
    }

    // Two-level reduction
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
