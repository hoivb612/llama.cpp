// mul_mat_vec_q3k_dp4a.hlsl - dp4a-accelerated Q3_K matvec (M=1)
//
// Uses dot4add_i8packed (SM 6.4) for integer dot products.
// Q3_K reconstructs each weight as q = (qs_2bit | (hbit << 2)) - 4, range
// [-4,3]. We keep q in [0,7] (unsigned) so it interprets as positive
// signed int8, then subtract the bias 4*sum(q8) at the end (mirrors the
// Q6_K -32 bias trick in mul_mat_vec_q6k_dp4a.hlsl).
//
// Processes 2 output rows per workgroup, sharing Q8_1 activation loads.
// src1 is pre-quantized Q8_1 data in a scratch buffer.
//
// Q3_K block (110 bytes, NOT 4-byte aligned):
//   hmask[32] + qs[64] + scales[12](16 packed 6-bit, biased -32) + d(f16)
// Q8_1 block (36 bytes): ds(2xf16 packed) + qs[32](int8 packed as 8 x uint32)
//
// 16 threads per Q3_K superblock — one thread per 16-element subblock
// (one scale value), 4 dp4a ops per thread covering 16 elements.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q3K_BSIZE   110
#define Q8_1_BSIZE  36
#define NUM_ROWS    2

// Wave-portable reduction LDS. See mul_mat_vec_q4k_dp4a.hlsl for rationale.
groupshared float shared_acc0[64];
groupshared float shared_acc1[64];

// Unaligned 4-byte load: Q3_K block stride is 110 bytes so block_off
// may not be 4-byte aligned. Always issues 2 aligned loads; relies on
// L1 cache to amortize when adjacent calls touch the same word.
uint load_u32_u(ByteAddressBuffer buf, uint byte_off) {
    uint align_off = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint w0 = buf.Load(align_off);
    if (shift == 0) return w0;
    uint w1 = buf.Load(align_off + 4);
    return (w0 >> shift) | (w1 << (32u - shift));
}

// Decode Q3_K data for one row's superblock for thread `t` (subblock 0..15).
// Outputs: d_super, signed 6-bit scale, and 4 packed unsigned int8 vectors
// (q in [0,7]; the -4 centering is applied at the end via psum).
void decode_q3k_row(uint block_off, uint t,
                    out float d_super, out int scale_signed,
                    out uint uq0, out uint uq1, out uint uq2, out uint uq3) {
    d_super = f16_to_f32(load_u32_u(src0, block_off + 108) & 0xFFFFu);

    // Per-subblock byte offsets (mirrors mul_mat_vec_q3k.hlsl).
    uint qs_off = ((t >> 3) & 1u) * 32u + (t & 1u) * 16u;
    uint hm_off = (t & 1u) * 16u;
    uint shift  = ((t >> 1) & 3u) * 2u;
    uint m_bit  = t >> 1;
    uint sub    = t >> 2;
    uint pos    = (t & 3u) * 8u;

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0F0F0F0Fu;

    // Decode this thread's 6-bit scale slot from the packed 12 bytes.
    uint scales_off = block_off + 96;
    uint raw0 = load_u32_u(src0, scales_off + 0);
    uint raw4 = load_u32_u(src0, scales_off + 4);
    uint raw8 = load_u32_u(src0, scales_off + 8);

    uint aux_word;
    if      (sub == 0u) aux_word = (raw0 & kmask2)         | (((raw8 >> 0u) & kmask1) << 4u);
    else if (sub == 1u) aux_word = (raw4 & kmask2)         | (((raw8 >> 2u) & kmask1) << 4u);
    else if (sub == 2u) aux_word = ((raw0 >> 4u) & kmask2) | (((raw8 >> 4u) & kmask1) << 4u);
    else                aux_word = ((raw4 >> 4u) & kmask2) | (((raw8 >> 6u) & kmask1) << 4u);
    scale_signed = int((aux_word >> pos) & 0xFFu) - 32;

    // 16 qs bytes (low 2 bits) and 16 hmask bytes (high bit) for this subblock.
    uint qs_block_off = block_off + 32u + qs_off;
    uint qw0 = load_u32_u(src0, qs_block_off + 0);
    uint qw1 = load_u32_u(src0, qs_block_off + 4);
    uint qw2 = load_u32_u(src0, qs_block_off + 8);
    uint qw3 = load_u32_u(src0, qs_block_off + 12);

    uint hm_block_off = block_off + 0u + hm_off;
    uint hw0 = load_u32_u(src0, hm_block_off + 0);
    uint hw1 = load_u32_u(src0, hm_block_off + 4);
    uint hw2 = load_u32_u(src0, hm_block_off + 8);
    uint hw3 = load_u32_u(src0, hm_block_off + 12);

    uint qp0 = (qw0 >> shift) & 0x03030303u;
    uint qp1 = (qw1 >> shift) & 0x03030303u;
    uint qp2 = (qw2 >> shift) & 0x03030303u;
    uint qp3 = (qw3 >> shift) & 0x03030303u;

    uint hp0 = (hw0 >> m_bit) & 0x01010101u;
    uint hp1 = (hw1 >> m_bit) & 0x01010101u;
    uint hp2 = (hw2 >> m_bit) & 0x01010101u;
    uint hp3 = (hw3 >> m_bit) & 0x01010101u;

    // q in [0,7]; bias correction (-4) applied at the end via psum.
    uq0 = qp0 | (hp0 << 2);
    uq1 = qp1 | (hp1 << 2);
    uq2 = qp2 | (hp2 << 2);
    uq3 = qp3 | (hp3 << 2);
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
    uint q8_vec_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_q8_per_vec * Q8_1_BSIZE;

    // 16 threads cooperate per Q3_K superblock (256 elems / 16 = 16 elems/thread)
    uint it_size = GROUP_SIZE / 16;
    uint itid = tid % 16;
    uint ix = tid / 16;

    // Q8_1 block: each thread covers half of one Q8_1 block (16 of 32 elems)
    uint q8_blk = itid / 2;
    uint q8_byte_off = 16u * (itid & 1u);  // 0 or 16

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint block_idx = ix; block_idx < num_blocks; block_idx += it_size) {
        // --- Load Q8_1 data (shared across both rows) ---
        uint q8_super_base = q8_vec_base + block_idx * 8u * Q8_1_BSIZE;
        uint q8_off = q8_super_base + q8_blk * Q8_1_BSIZE;

        uint ds = src1.Load(q8_off);
        float q8d = f16_to_f32(ds & 0xFFFFu);

        uint q8_qs0 = src1.Load(q8_off + 4 + q8_byte_off + 0);
        uint q8_qs1 = src1.Load(q8_off + 4 + q8_byte_off + 4);
        uint q8_qs2 = src1.Load(q8_off + 4 + q8_byte_off + 8);
        uint q8_qs3 = src1.Load(q8_off + 4 + q8_byte_off + 12);

        // psum: sum of 16 Q8 bytes (used for the -4 bias correction).
        // Use separate zero-initialized dot4add accumulators (as in
        // mul_mat_vec_q4k_dp4a.hlsl) rather than chaining into one running
        // accumulator with a constant first operand -- the chained-constant
        // form is miscompiled on some drivers and yields a wrong sum.
        int p0 = 0; p0 = dot4add_i8packed(0x01010101u, q8_qs0, p0);
        int p1 = 0; p1 = dot4add_i8packed(0x01010101u, q8_qs1, p1);
        int p2 = 0; p2 = dot4add_i8packed(0x01010101u, q8_qs2, p2);
        int p3 = 0; p3 = dot4add_i8packed(0x01010101u, q8_qs3, p3);
        int q8_psum = p0 + p1 + p2 + p3;

        // --- Row 0 ---
        {
            float d_super; int scale_signed;
            uint uq0, uq1, uq2, uq3;
            decode_q3k_row(src0_row0 + block_idx * Q3K_BSIZE, itid,
                           d_super, scale_signed, uq0, uq1, uq2, uq3);

            int isx = 0;
            isx = dot4add_i8packed(uq0, q8_qs0, isx);
            isx = dot4add_i8packed(uq1, q8_qs1, isx);
            isx = dot4add_i8packed(uq2, q8_qs2, isx);
            isx = dot4add_i8packed(uq3, q8_qs3, isx);

            float scale_f = d_super * float(scale_signed) * q8d;
            acc0 = mad(scale_f, float(isx - 4 * q8_psum), acc0);
        }

        // --- Row 1 ---
        {
            float d_super; int scale_signed;
            uint uq0, uq1, uq2, uq3;
            decode_q3k_row(src0_row1 + block_idx * Q3K_BSIZE, itid,
                           d_super, scale_signed, uq0, uq1, uq2, uq3);

            int isx = 0;
            isx = dot4add_i8packed(uq0, q8_qs0, isx);
            isx = dot4add_i8packed(uq1, q8_qs1, isx);
            isx = dot4add_i8packed(uq2, q8_qs2, isx);
            isx = dot4add_i8packed(uq3, q8_qs3, isx);

            float scale_f = d_super * float(scale_signed) * q8d;
            acc1 = mad(scale_f, float(isx - 4 * q8_psum), acc1);
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
