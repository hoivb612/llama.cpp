// mul_mat_vec_glu_q4_k.hlsl - Fused MUL_MAT(M=1, W_gate Q4_K) + MUL_MAT(M=1, W_up Q4_K) + GLU(SwiGLU split)
//
// R9 fusion variant for Q4_K weights. Computes:
//   t_gate = W_gate @ x          (Q4_K matvec)
//   t_up   = W_up   @ x          (Q4_K matvec, same x, same shape as W_gate)
//   y      = silu(t_gate) * t_up (split-mode SwiGLU)
//
// Activations are loaded once per group iteration and reused across the
// gate-row0 / gate-row1 / up-row0 / up-row1 dot products.
//
// Q4_K layout per superblock (Q4K_BSIZE = 144 bytes):
//   [0..3]   d (f16) | dmin (f16)
//   [4..15]  12 bytes packed 6-bit (scale, min) for 8 sub-blocks of 32 elements
//   [16..]   128 bytes of 4-bit quantized values
//
// Phase A: cooperative LDS pre-decode of (dall * sub_scale[i], dmin * sub_min[i])
// for i in 0..7 per role per block. Avoids redundant scale decoding across the
// 16 threads that share each block in Phase B.
//
// Phase B: GROUP_SIZE = 64, 16 threads per Q4_K block, 4 blocks per iteration.
// Each thread processes 16 elements (4 elements per sub-block, across 4 of
// the 8 sub-blocks). The activation segment loaded by each thread covers the
// 16 elements (4 from each of by0..by15).
//
// Phase C: 4-way reduction over the workgroup, SiLU(gate)*up, write 2 outputs.
//
// LDS budget: 4 (roles) * MAX_BLOCKS * 16 (8 scales + 8 mins) floats. With
// MAX_BLOCKS = 16 this is 1024 floats = 4 KB. Covers K up to 4096; the
// dispatch path falls back to the unfused matvec above that threshold.
//
// RMS_FUSED variant (mul_mat_vec_glu_q4_k_rms.hlsl): folds the preceding
// RMS_NORM + MUL(norm_weight) in. src1 carries the pre-norm activation x and
// src6 the norm weight g; every term (including the Q4_K min correction) is
// linear in the activation, so one pass accumulates the dots against x*g plus
// sum(x*x) and applies 1/rms once at the end. op14 = eps (float bits).

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE   64
#endif
// 5 partials (gate0/up0/gate1/up1/sum-of-squares) per wave. The smallest wave
// this backend targets is 16 lanes, so GROUP_SIZE/16 bounds the wave count.
#define MAX_WAVES    (GROUP_SIZE / 16 > 0 ? GROUP_SIZE / 16 : 1)
#define QK_K         256
#define Q4K_BSIZE    144
#define MAX_BLOCKS   16    // K up to 4096; falls back to unfused above this

groupshared float ld_scales[MAX_BLOCKS * 4 * 8];   // dall * sub_scale[i]
groupshared float ld_mins  [MAX_BLOCKS * 4 * 8];   // dmin * sub_min[i]
groupshared float shared_acc[5 * MAX_WAVES];

// Decode one Q4_K block header: 4 bytes (d, dmin) + 12 bytes packed 6-bit
// (scale, min) for 8 sub-blocks. Writes the 8 (dall * sub_scale) and 8
// (dmin * sub_min) products to LDS at base_idx.
void store_q4k_header(ByteAddressBuffer buf, uint block_off, uint base_idx) {
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

    // Sub-block scale[i]: bytes[0..3] low 6 bits, then bytes[8..11] low 4 bits
    // | top 2 bits of bytes[0..3].
    // Sub-block min  [i]: bytes[4..7] low 6 bits, then bytes[8..11] high 4 bits
    // | top 2 bits of bytes[4..7].
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

// Per-thread, per-block dequant + dot. Reads 32 bytes of qs, produces the
// (sx, sy, sz, sw) partials that pair with sub-block scales (sg+0, sg+1,
// sg+4, sg+5).
void q4k_block_dot(ByteAddressBuffer buf, uint qs_off, uint q_offset,
                   float by0, float by1, float by2, float by3,
                   float by4, float by5, float by6, float by7,
                   float by8, float by9, float by10, float by11,
                   float by12, float by13, float by14, float by15,
                   out float sx, out float sy, out float sz, out float sw) {
    uint qs0  = buf.Load(qs_off + q_offset);
    uint qs64 = buf.Load(qs_off + q_offset + 64);
    uint qs0_lo  = qs0 & 0x0F0F0F0Fu;
    uint qs0_hi  = (qs0 >> 4) & 0x0F0F0F0Fu;
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
            store_q4k_header(src0, gate_row0 + blk * Q4K_BSIZE, base_idx);
        } else if (role == 1u) {
            store_q4k_header(src0, gate_row1 + blk * Q4K_BSIZE, base_idx);
        } else if (role == 2u) {
            store_q4k_header(src2, up_row0   + blk * Q4K_BSIZE, base_idx);
        } else {
            store_q4k_header(src2, up_row1   + blk * Q4K_BSIZE, base_idx);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Phase B: 16 threads per Q4_K block, 4 blocks per iter ----
    uint it_size = GROUP_SIZE / 16;          // = 4 blocks per iter
    uint itid    = tid % 16;
    uint ix      = tid / 16;

    uint il   = itid / 4;        // 0..3
    uint ir   = itid % 4;        // 0..3
    uint v_im = il / 2;          // 0 or 1: which 128-element half
    uint v_in = il % 2;          // 0 or 1
    uint sg   = v_im * 2u;       // sub-block index "a" within this half: 0 or 2

    uint l0       = 4 * (2 * ir + v_in);
    uint q_offset = 32 * v_im + l0;
    uint y_offset = 64 * v_im + l0;

    float acc_g0 = 0.0f, acc_g1 = 0.0f, acc_u0 = 0.0f, acc_u1 = 0.0f;
#if RMS_FUSED
    float acc_ss = 0.0f;
#endif

    for (uint blk = ix; blk < num_blocks; blk += it_size) {
        // Activation: 16 floats, shared across all 4 roles.
        uint y1_off = src1_base + (blk * QK_K + y_offset) * 4;
        uint y2_off = y1_off + 128 * 4;
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
        acc_ss += by0*by0 + by1*by1 + by2*by2 + by3*by3
                + by4*by4 + by5*by5 + by6*by6 + by7*by7
                + by8*by8 + by9*by9 + by10*by10 + by11*by11
                + by12*by12 + by13*by13 + by14*by14 + by15*by15;
        {
            uint g1_off = (blk * QK_K + y_offset) * 4;
            uint g2_off = g1_off + 128 * 4;
            float4 g0 = asfloat(src6.Load4(g1_off));
            float4 g1 = asfloat(src6.Load4(g1_off + 128));
            float4 g2 = asfloat(src6.Load4(g2_off));
            float4 g3 = asfloat(src6.Load4(g2_off + 128));
            by0  *= g0.x; by1  *= g0.y; by2  *= g0.z; by3  *= g0.w;
            by4  *= g1.x; by5  *= g1.y; by6  *= g1.z; by7  *= g1.w;
            by8  *= g2.x; by9  *= g2.y; by10 *= g2.z; by11 *= g2.w;
            by12 *= g3.x; by13 *= g3.y; by14 *= g3.z; by15 *= g3.w;
        }
#endif

        float sum_a = by0  + by1  + by2  + by3;
        float sum_b = by4  + by5  + by6  + by7;
        float sum_c = by8  + by9  + by10 + by11;
        float sum_d = by12 + by13 + by14 + by15;

        // Iterate 4 roles; each reads pre-decoded scales/mins from LDS.
        // Roles 0,1 use src0 (gate weights); roles 2,3 use src2 (up weights).
        // Helper macro to keep this short.
        #define DO_ROLE(BUF, ROW_BASE, ROLE, ACC)                                        \
        {                                                                                \
            uint blk_off = (ROW_BASE) + blk * Q4K_BSIZE;                                 \
            float sx, sy, sz, sw;                                                        \
            q4k_block_dot(BUF, blk_off + 16, q_offset,                                   \
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
#if RMS_FUSED
    float wave_ss = WaveActiveSum(acc_ss);
#endif

    uint wave_id   = tid / WARP_SIZE;
    uint num_waves = (GROUP_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[                wave_id] = wave_g0;
        shared_acc[num_waves     + wave_id] = wave_u0;
        shared_acc[num_waves * 2 + wave_id] = wave_g1;
        shared_acc[num_waves * 3 + wave_id] = wave_u1;
#if RMS_FUSED
        shared_acc[num_waves * 4 + wave_id] = wave_ss;
#endif
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[              tid] += shared_acc[              tid + s];
            shared_acc[num_waves   + tid] += shared_acc[num_waves   + tid + s];
            shared_acc[num_waves*2 + tid] += shared_acc[num_waves*2 + tid + s];
            shared_acc[num_waves*3 + tid] += shared_acc[num_waves*3 + tid + s];
#if RMS_FUSED
            shared_acc[num_waves*4 + tid] += shared_acc[num_waves*4 + tid + s];
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float gate0 = shared_acc[0];
        float up0   = shared_acc[num_waves];
#if RMS_FUSED
        float rms_scale = 1.0f / sqrt(shared_acc[num_waves * 4] / (float)K + asfloat(op14));
        gate0 *= rms_scale;
        up0   *= rms_scale;
#endif
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float gate1 = shared_acc[num_waves * 2];
            float up1   = shared_acc[num_waves * 3];
#if RMS_FUSED
            gate1 *= rms_scale;
            up1   *= rms_scale;
#endif
            float result1 = (gate1 / (1.0f + exp(-gate1))) * up1;
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
