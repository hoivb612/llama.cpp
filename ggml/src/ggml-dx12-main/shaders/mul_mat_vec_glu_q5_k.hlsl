// mul_mat_vec_glu_q5_k.hlsl - Fused MUL_MAT(M=1, W_gate Q5_K) + MUL_MAT(M=1, W_up Q5_K) + GLU(SwiGLU split)
//
// R9 fusion variant for Q5_K weights. Same structure as the Q4_K fusion shader
// (cooperative LDS pre-decode of scales / mins, 4-role inner loop, SwiGLU
// reduction), with a different qs/qh access pattern that mirrors
// mul_mat_vec_q5k_mr.hlsl byte-for-byte.
//
// Q5_K layout per superblock (Q5K_BSIZE = 176 bytes):
//   [0..3]    d (f16) | dmin (f16)
//   [4..15]   12 bytes packed 6-bit (scale, min) for 8 sub-blocks of 32 elements
//   [16..47]  qh: 32 bytes of 5th bits
//   [48..175] 128 bytes of 4-bit quantized values
//
// Phase A: cooperative LDS pre-decode of (dall*sub_scale[i], dmin*sub_min[i])
// for i in 0..7 per role per block. Same algorithm as Q4_K fusion; the header
// bytes 0..15 are identical to Q4_K so we reuse the decode logic.
//
// Phase B: GROUP_SIZE = 64, 16 threads per Q5_K block, 4 blocks per iteration.
// Each thread processes 16 elements via the Load2-based stride pattern from
// the canonical Q5_K MR shader, OR-ing the 5th bit from qh into each nibble
// before the dot product.
//
// Phase C: 4-way reduction over the workgroup, SiLU(gate) * up, write 2 outputs.
//
// LDS budget: 4 (roles) * MAX_BLOCKS * 16 (8 scales + 8 mins) floats. With
// MAX_BLOCKS = 16 this is 1024 floats = 4 KB. Covers K up to 4096; the dispatch
// path falls back to the unfused matvec above that threshold.

#include "ggml_common.hlsli"

#define GROUP_SIZE   64
#define QK_K         256
#define Q5K_BSIZE    176
#define MAX_BLOCKS   16    // K up to 4096; falls back to unfused above this

groupshared float ld_scales[MAX_BLOCKS * 4 * 8];   // dall * sub_scale[i]
groupshared float ld_mins  [MAX_BLOCKS * 4 * 8];   // dmin * sub_min[i]
groupshared float shared_acc[16];                  // 4 partials × max 4 waves

// Q5_K row stride is 176 bytes per block (4-aligned), so byte_addr at qh+l0 or
// qs+q_offset can be 2-aligned. Mirror Q5_K MR's safe 16-bit loader.
uint safe_load_u16(ByteAddressBuffer buf, uint byte_addr) {
    uint aligned = byte_addr & ~3u;
    uint raw = buf.Load(aligned);
    return ((byte_addr & 2u) != 0u) ? ((raw >> 16) & 0xFFFFu) : (raw & 0xFFFFu);
}

// Decode one Q5_K block header: 4 bytes (d, dmin) + 12 bytes packed 6-bit
// (scale, min) for 8 sub-blocks. Writes the 8 (dall * sub_scale) and 8
// (dmin * sub_min) products to LDS at base_idx.
//
// Header bytes are identical to Q4_K, so this function is structurally a
// copy of store_q4k_header.
void store_q5k_header(ByteAddressBuffer buf, uint block_off, uint base_idx) {
    uint dm_raw = buf.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin = f16_to_f32(dm_raw >> 16);

    uint s0 = buf.Load(block_off + 4);    // bytes 0..3
    uint s1 = buf.Load(block_off + 8);    // bytes 4..7
    uint s2 = buf.Load(block_off + 12);   // bytes 8..11

    uint b0 = (s0      ) & 0xFFu;
    uint b1 = (s0 >>  8) & 0xFFu;
    uint b2 = (s0 >> 16) & 0xFFu;
    uint b3 = (s0 >> 24) & 0xFFu;
    uint b4 = (s1      ) & 0xFFu;
    uint b5 = (s1 >>  8) & 0xFFu;
    uint b6 = (s1 >> 16) & 0xFFu;
    uint b7 = (s1 >> 24) & 0xFFu;
    uint b8  = (s2      ) & 0xFFu;
    uint b9  = (s2 >>  8) & 0xFFu;
    uint b10 = (s2 >> 16) & 0xFFu;
    uint b11 = (s2 >> 24) & 0xFFu;

    uint sc[8];
    uint mn[8];
    sc[0] = b0 & 0x3Fu;
    sc[1] = b1 & 0x3Fu;
    sc[2] = b2 & 0x3Fu;
    sc[3] = b3 & 0x3Fu;
    mn[0] = b4 & 0x3Fu;
    mn[1] = b5 & 0x3Fu;
    mn[2] = b6 & 0x3Fu;
    mn[3] = b7 & 0x3Fu;
    sc[4] = (b8  & 0x0Fu) | ((b0 >> 2) & 0x30u);
    sc[5] = (b9  & 0x0Fu) | ((b1 >> 2) & 0x30u);
    sc[6] = (b10 & 0x0Fu) | ((b2 >> 2) & 0x30u);
    sc[7] = (b11 & 0x0Fu) | ((b3 >> 2) & 0x30u);
    mn[4] = (b8  >> 4) | ((b4 >> 2) & 0x30u);
    mn[5] = (b9  >> 4) | ((b5 >> 2) & 0x30u);
    mn[6] = (b10 >> 4) | ((b6 >> 2) & 0x30u);
    mn[7] = (b11 >> 4) | ((b7 >> 2) & 0x30u);

    [unroll] for (uint i = 0; i < 8; ++i) {
        ld_scales[base_idx + i] = dall * float(sc[i]);
        ld_mins  [base_idx + i] = dmin * float(mn[i]);
    }
}

// Per-thread, per-block dequant + dot for Q5_K. Reads qs / qh as Q5_K MR does
// (Load2-based pattern), and produces (sx, sy, sz, sw) partials that pair with
// sub-block scales (sg+0, sg+1, sg+4, sg+5).
void q5k_block_dot(ByteAddressBuffer buf, uint block_off,
                   uint q_offset, uint qh_off_l0, uint qh_shift,
                   float by0, float by1, float by2, float by3,
                   float by4, float by5, float by6, float by7,
                   float by8, float by9, float by10, float by11,
                   float by12, float by13, float by14, float by15,
                   out float sx, out float sy, out float sz, out float sw) {
    // qs at block_off + 48
    uint qs_off = block_off + 48;
    uint qs_lo0  = safe_load_u16(buf, qs_off + q_offset);
    uint qs_lo16 = safe_load_u16(buf, qs_off + q_offset + 16);
    uint qs0_16_u32 = qs_lo0 | (qs_lo16 << 16);

    uint qs_lo64 = safe_load_u16(buf, qs_off + q_offset + 64);
    uint qs_lo80 = safe_load_u16(buf, qs_off + q_offset + 80);
    uint qs64_80_u32 = qs_lo64 | (qs_lo80 << 16);

    uint qs0_lo4  = qs0_16_u32 & 0x0F0F0F0Fu;
    uint qs0_hi4  = (qs0_16_u32 >> 4) & 0x0F0F0F0Fu;
    uint qs64_lo4 = qs64_80_u32 & 0x0F0F0F0Fu;
    uint qs64_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0Fu;

    // qh at block_off + 16, accessed via two safe_load_u16 calls 16 bytes apart
    uint qh_off = block_off + 16;
    uint qh_lo0  = safe_load_u16(buf, qh_off + qh_off_l0);
    uint qh_lo16 = safe_load_u16(buf, qh_off + qh_off_l0 + 16);
    uint qh = qh_lo0 | (qh_lo16 << 16);

    qs0_lo4  += ((qh >> qh_shift) & 0x01010101u) << 4;
    qs0_hi4  += ((qh >> qh_shift) & 0x02020202u) << 3;
    qs64_lo4 += (qh >> qh_shift) & 0x10101010u;
    qs64_hi4 += ((qh >> qh_shift) & 0x20202020u) >> 1;

    float q0  = float(qs0_lo4 & 0xFFu);
    float q1  = float((qs0_lo4 >> 8) & 0xFFu);
    float q2  = float((qs0_lo4 >> 16) & 0xFFu);
    float q3  = float(qs0_lo4 >> 24);
    float q4  = float(qs0_hi4 & 0xFFu);
    float q5  = float((qs0_hi4 >> 8) & 0xFFu);
    float q6  = float((qs0_hi4 >> 16) & 0xFFu);
    float q7  = float(qs0_hi4 >> 24);
    float q8  = float(qs64_lo4 & 0xFFu);
    float q9  = float((qs64_lo4 >> 8) & 0xFFu);
    float q10 = float((qs64_lo4 >> 16) & 0xFFu);
    float q11 = float(qs64_lo4 >> 24);
    float q12 = float(qs64_hi4 & 0xFFu);
    float q13 = float((qs64_hi4 >> 8) & 0xFFu);
    float q14 = float((qs64_hi4 >> 16) & 0xFFu);
    float q15 = float(qs64_hi4 >> 24);

    sx = mad(q0,  by0,  mad(q1,  by1,  mad(q2,  by2,  q3  * by3 )));
    sy = mad(q4,  by4,  mad(q5,  by5,  mad(q6,  by6,  q7  * by7 )));
    sz = mad(q8,  by8,  mad(q9,  by9,  mad(q10, by10, q11 * by11)));
    sw = mad(q12, by12, mad(q13, by13, mad(q14, by14, q15 * by15)));
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint row1 = min(row0 + 1, ne0 - 1);

    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K          = ne00;
    uint num_blocks = K / QK_K;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint up_base   = op1          + i2_src0 * nb02 + i3_src0 * nb03;
    uint gate_row0 = src0_base + row0 * nb01;
    uint gate_row1 = src0_base + row1 * nb01;
    uint up_row0   = up_base   + row0 * nb01;
    uint up_row1   = up_base   + row1 * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    // ---- Phase A: cooperative scale pre-decode ----
    uint total_entries = 4u * num_blocks;
    for (uint e = tid; e < total_entries; e += GROUP_SIZE) {
        uint role = e & 3u;        // 0=g0, 1=g1, 2=u0, 3=u1
        uint blk  = e >> 2;
        uint base_idx = (blk * 4u + role) * 8u;
        if (role == 0u) {
            store_q5k_header(src0, gate_row0 + blk * Q5K_BSIZE, base_idx);
        } else if (role == 1u) {
            store_q5k_header(src0, gate_row1 + blk * Q5K_BSIZE, base_idx);
        } else if (role == 2u) {
            store_q5k_header(src2, up_row0   + blk * Q5K_BSIZE, base_idx);
        } else {
            store_q5k_header(src2, up_row1   + blk * Q5K_BSIZE, base_idx);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Phase B: 16 threads per Q5_K block, 4 blocks per iter ----
    uint it_size = GROUP_SIZE / 16;          // = 4 blocks per iter
    uint itid    = tid % 16;
    uint ix      = tid / 16;

    uint il   = itid / 4;        // 0..3
    uint ir   = itid % 4;        // 0..3
    uint v_im = il / 2;          // 0 or 1: which 128-element half
    uint v_in = il % 2;          // 0 or 1
    uint sg   = v_im * 2u;       // sub-block index "a" within this half: 0 or 2

    // Q5_K MR's stride-2 thread layout (different from Q4_K MR's stride-4).
    uint l0       = 4 * ir + 2 * v_in;
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 64 * v_im + l0;
    uint qh_shift = 2 * v_im;

    float acc_g0 = 0.0f, acc_g1 = 0.0f, acc_u0 = 0.0f, acc_u1 = 0.0f;

    for (uint blk = ix; blk < num_blocks; blk += it_size) {
        // Activation: 16 floats (Q5_K MR layout uses Load2 across stride-2 positions).
        uint y1_off = src1_base + (blk * QK_K + y_offset) * 4;
        uint y2_off = y1_off + 128 * 4;

        uint2 p01 = src1.Load2(y1_off);
        uint2 p23 = src1.Load2(y1_off + 64);
        uint2 p45 = src1.Load2(y1_off + 128);
        uint2 p67 = src1.Load2(y1_off + 192);
        uint2 p89 = src1.Load2(y2_off);
        uint2 pab = src1.Load2(y2_off + 64);
        uint2 pcd = src1.Load2(y2_off + 128);
        uint2 pef = src1.Load2(y2_off + 192);

        float by0  = asfloat(p01.x); float by1  = asfloat(p01.y);
        float by2  = asfloat(p23.x); float by3  = asfloat(p23.y);
        float by4  = asfloat(p45.x); float by5  = asfloat(p45.y);
        float by6  = asfloat(p67.x); float by7  = asfloat(p67.y);
        float by8  = asfloat(p89.x); float by9  = asfloat(p89.y);
        float by10 = asfloat(pab.x); float by11 = asfloat(pab.y);
        float by12 = asfloat(pcd.x); float by13 = asfloat(pcd.y);
        float by14 = asfloat(pef.x); float by15 = asfloat(pef.y);

        float sum_a = by0  + by1  + by2  + by3;
        float sum_b = by4  + by5  + by6  + by7;
        float sum_c = by8  + by9  + by10 + by11;
        float sum_d = by12 + by13 + by14 + by15;

        #define DO_ROLE(BUF, ROW_BASE, ROLE, ACC)                                        \
        {                                                                                \
            uint blk_off = (ROW_BASE) + blk * Q5K_BSIZE;                                 \
            float sx, sy, sz, sw;                                                        \
            q5k_block_dot(BUF, blk_off, q_offset, l0, qh_shift,                          \
                          by0,  by1,  by2,  by3,                                         \
                          by4,  by5,  by6,  by7,                                         \
                          by8,  by9,  by10, by11,                                        \
                          by12, by13, by14, by15,                                        \
                          sx, sy, sz, sw);                                               \
            uint base = (blk * 4u + (ROLE)) * 8u;                                        \
            float sc0 = ld_scales[base + sg + 0u];                                       \
            float sc1 = ld_scales[base + sg + 1u];                                       \
            float sc4 = ld_scales[base + sg + 4u];                                       \
            float sc5 = ld_scales[base + sg + 5u];                                       \
            float m_a = ld_mins  [base + sg + 0u];                                       \
            float m_b = ld_mins  [base + sg + 1u];                                       \
            float m_c = ld_mins  [base + sg + 4u];                                       \
            float m_d = ld_mins  [base + sg + 5u];                                       \
            float smin = m_a * sum_a + m_b * sum_b + m_c * sum_c + m_d * sum_d;          \
            (ACC) += (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - smin;                 \
        }

        DO_ROLE(src0, gate_row0, 0u, acc_g0);
        DO_ROLE(src0, gate_row1, 1u, acc_g1);
        DO_ROLE(src2, up_row0,   2u, acc_u0);
        DO_ROLE(src2, up_row1,   3u, acc_u1);

        #undef DO_ROLE
    }

    // ---- Phase C: cross-wave reduce + SwiGLU + write ----
    float wave_g0 = WaveActiveSum(acc_g0);
    float wave_g1 = WaveActiveSum(acc_g1);
    float wave_u0 = WaveActiveSum(acc_u0);
    float wave_u1 = WaveActiveSum(acc_u1);

    uint wave_id   = tid / WARP_SIZE;
    uint num_waves = (GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[                wave_id] = wave_g0;
        shared_acc[num_waves     + wave_id] = wave_u0;
        shared_acc[num_waves * 2 + wave_id] = wave_g1;
        shared_acc[num_waves * 3 + wave_id] = wave_u1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[              tid] += shared_acc[              tid + s];
            shared_acc[num_waves   + tid] += shared_acc[num_waves   + tid + s];
            shared_acc[num_waves*2 + tid] += shared_acc[num_waves*2 + tid + s];
            shared_acc[num_waves*3 + tid] += shared_acc[num_waves*3 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float gate0 = shared_acc[0];
        float up0   = shared_acc[num_waves];
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float gate1 = shared_acc[num_waves * 2];
            float up1   = shared_acc[num_waves * 3];
            float result1 = (gate1 / (1.0f + exp(-gate1))) * up1;
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
