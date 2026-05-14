// mul_mat_vec_q3k_mr_blocked.hlsl - Block-level Q3_K matvec (M=1, 2 rows/group)
//
// Q3_K block (110 bytes per 256 elements, NOT 4-byte aligned):
//   offset   0..31  : hmask[32] (1 bit per element, high bit of 3-bit quant)
//   offset  32..95  : qs[64]    (2 bits per element, low bits)
//   offset  96..107 : scales[12] (16 packed 6-bit values, biased by 32)
//   offset 108..109 : d (fp16)
//
// 256 threads, 16 threads per Q3_K block, 16 elements per thread.
// Mirrors the Q4_K/Q5_K/Q6_K MR block-level layout: each lane decodes a single
// scale group (16 contiguous elements) with a vectorised qs Load4 + hmask
// Load4, sharing one activation Load4 stream across two output rows.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch * ne2 * ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q3K_BSIZE   110
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

    // Thread mapping: 16 threads per Q3_K block (one per scale group)
    uint it_size = GROUP_SIZE / 16;
    uint scale_idx = tid % 16;
    uint ix = tid / 16;

    // Per-thread element parameters (constant across the block loop)
    uint n      = scale_idx >> 3;             // 0 or 1 (half-block)
    uint j_h    = scale_idx & 7u;             // 0..7 (group within half)
    uint j      = j_h >> 1;                   // 0..3
    uint half_  = j_h & 1u;                   // 0 or 1
    uint shift  = j << 1;                     // qs shift: 0,2,4,6
    uint m_bit  = (n << 2) + j;               // hmask bit: 0..7
    uint qs_chunk_off = 32u + (n << 5) + (half_ << 4); // 16 contiguous qs bytes
    uint hm_chunk_off = (half_ << 4);                  // 16 contiguous hmask bytes
    uint y_offset = scale_idx * 16;                    // 16 contiguous floats

    // Per-thread scale unpack constants (scales[16] packed in 12 bytes)
    uint sub = scale_idx >> 2;            // 0..3
    uint pos_byte = scale_idx & 3u;       // 0..3
    uint base_sel = (sub & 1u) ? 4u : 0u; // 0 -> bytes 96..99, 4 -> bytes 100..103
    uint hi_shift = sub << 1;             // 0,2,4,6

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

        // --- Process both rows ---
        [unroll]
        for (uint r = 0; r < NUM_ROWS; r++) {
            uint block_off = (r == 0u) ? (src0_row0 + block_idx * Q3K_BSIZE)
                                       : (src0_row1 + block_idx * Q3K_BSIZE);

            // d at offsets 108..109; assemble from bytes
            uint d_lo_word = src0.Load((block_off + 108u) & ~3u);
            uint d_lo_byte = (d_lo_word >> (((block_off + 108u) & 3u) * 8u)) & 0xFFu;
            uint d_hi_word = src0.Load((block_off + 109u) & ~3u);
            uint d_hi_byte = (d_hi_word >> (((block_off + 109u) & 3u) * 8u)) & 0xFFu;
            float d = f16_to_f32(d_lo_byte | (d_hi_byte << 8));

            // Decode this thread's signed 6-bit scale (biased by 32)
            uint scales_off = block_off + 96u;
            uint base_off = scales_off + base_sel + pos_byte;
            uint base_w   = src0.Load(base_off & ~3u);
            uint base_b   = (base_w >> ((base_off & 3u) * 8u)) & 0xFFu;
            uint hi_off   = scales_off + 8u + pos_byte;
            uint hi_w     = src0.Load(hi_off & ~3u);
            uint hi_b     = (hi_w >> ((hi_off & 3u) * 8u)) & 0xFFu;
            uint base_nib = (sub < 2u) ? (base_b & 0x0Fu) : (base_b >> 4);
            uint hi_nib   = (hi_b >> hi_shift) & 0x03u;
            int  scale_signed = int(base_nib | (hi_nib << 4)) - 32;
            float scale_d = d * float(scale_signed);

            // qs: 16 contiguous bytes for this scale group
            uint4 qs4 = src0.Load4(block_off + qs_chunk_off);
            // hmask: 16 contiguous bytes (same region for both n; differs by m_bit)
            uint4 hm4 = src0.Load4(block_off + hm_chunk_off);

            // Extract per-element 2-bit qs values (shift = j*2, mask 0x03)
            uint4 q_lo4 = (qs4 >> shift) & 0x03030303u;
            // Extract per-element hmask bit at m_bit position
            uint4 hm_sel4 = (hm4 >> m_bit) & 0x01010101u;

            // q = q_lo - (hbit ? 0 : 4) = q_lo + 4*hbit - 4
            // Compute per element as float and dot with activations.
            float dot_q = 0.0f;
            {
                int q0  = int( q_lo4.x        & 0xFFu) - 4 + (int)((hm_sel4.x        & 0xFFu) << 2);
                int q1  = int((q_lo4.x >> 8 ) & 0xFFu) - 4 + (int)(((hm_sel4.x >> 8 ) & 0xFFu) << 2);
                int q2  = int((q_lo4.x >> 16) & 0xFFu) - 4 + (int)(((hm_sel4.x >> 16) & 0xFFu) << 2);
                int q3  = int( q_lo4.x >> 24)         - 4 + (int)((hm_sel4.x >> 24) << 2);
                int q4  = int( q_lo4.y        & 0xFFu) - 4 + (int)((hm_sel4.y        & 0xFFu) << 2);
                int q5  = int((q_lo4.y >> 8 ) & 0xFFu) - 4 + (int)(((hm_sel4.y >> 8 ) & 0xFFu) << 2);
                int q6  = int((q_lo4.y >> 16) & 0xFFu) - 4 + (int)(((hm_sel4.y >> 16) & 0xFFu) << 2);
                int q7  = int( q_lo4.y >> 24)         - 4 + (int)((hm_sel4.y >> 24) << 2);
                int q8  = int( q_lo4.z        & 0xFFu) - 4 + (int)((hm_sel4.z        & 0xFFu) << 2);
                int q9  = int((q_lo4.z >> 8 ) & 0xFFu) - 4 + (int)(((hm_sel4.z >> 8 ) & 0xFFu) << 2);
                int q10 = int((q_lo4.z >> 16) & 0xFFu) - 4 + (int)(((hm_sel4.z >> 16) & 0xFFu) << 2);
                int q11 = int( q_lo4.z >> 24)         - 4 + (int)((hm_sel4.z >> 24) << 2);
                int q12 = int( q_lo4.w        & 0xFFu) - 4 + (int)((hm_sel4.w        & 0xFFu) << 2);
                int q13 = int((q_lo4.w >> 8 ) & 0xFFu) - 4 + (int)(((hm_sel4.w >> 8 ) & 0xFFu) << 2);
                int q14 = int((q_lo4.w >> 16) & 0xFFu) - 4 + (int)(((hm_sel4.w >> 16) & 0xFFu) << 2);
                int q15 = int( q_lo4.w >> 24)         - 4 + (int)((hm_sel4.w >> 24) << 2);

                float s0 = mad(float(q0),  by0,  mad(float(q1),  by1,  mad(float(q2),  by2,  float(q3)  * by3 )));
                float s1 = mad(float(q4),  by4,  mad(float(q5),  by5,  mad(float(q6),  by6,  float(q7)  * by7 )));
                float s2 = mad(float(q8),  by8,  mad(float(q9),  by9,  mad(float(q10), by10, float(q11) * by11)));
                float s3 = mad(float(q12), by12, mad(float(q13), by13, mad(float(q14), by14, float(q15) * by15)));
                dot_q = (s0 + s1) + (s2 + s3);
            }

            float row_contrib = scale_d * dot_q;
            if (r == 0u) acc0 += row_contrib;
            else         acc1 += row_contrib;
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
