// mul_mat_vec_q3k.hlsl - Optimized matrix-vector multiply for Q3_K weights (M=1)
//
// Q3_K block (110 bytes — NOT 4-byte aligned):
//   offset   0..31 : hmask[32] (1 bit per element, high bit of 3-bit quant)
//   offset  32..95 : qs[64]    (2 bits per element, low bits)
//   offset  96..107: scales[12] (16 packed 6-bit values, biased; subtract 32)
//   offset 108..109: d (fp16)
//
// Because Q3K_BSIZE=110 is not a multiple of 4, src0.Load() at block_off+offset is
// only 4-byte aligned for every other block. Use load_u32_q3k() helper which
// handles arbitrary alignment via two aligned word reads + shift.
//
// 256 threads, 16 cooperating per superblock. 1 output row per group.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE  256
#endif
#define QK_K        256
#define Q3K_BSIZE   110

groupshared float shared_acc[GROUP_SIZE];

// Aligned-safe 32-bit load from a byte offset of arbitrary alignment.
// On Intel/AMD plain Load() of unaligned addresses returns the *aligned*
// word (the &~3 mask is implicit in HW), giving wrong bytes. NVIDIA simply
// returns garbage. Always go through this helper for Q3_K.
uint load_u32_q3k(ByteAddressBuffer buf, uint addr) {
    uint a = addr & ~3u;
    uint shift = (addr & 3u) * 8u;
    uint w0 = buf.Load(a);
    if (shift == 0u) return w0;
    uint w1 = buf.Load(a + 4u);
    return (w0 >> shift) | (w1 << (32u - shift));
}

uint load_u16_q3k(ByteAddressBuffer buf, uint addr) {
    return load_u32_q3k(buf, addr) & 0xFFFFu;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_x_2d(group_id);
    if (i0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK_K;

    uint src0_row  = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    uint it_size = GROUP_SIZE / 16;
    uint is = tid % 16;
    uint ix = tid / 16;

    uint qs_off  = ((is >> 3) & 1u) * 32u + (is & 1u) * 16u;
    uint hm_off  = (is & 1u) * 16u;
    uint shift   = ((is >> 1) & 3u) * 2u;
    uint m_bit   = is >> 1;
    uint sub     = is >> 2;
    uint pos     = (is & 3u) * 8u;

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0F0F0F0Fu;

    float acc = 0.0f;

    for (uint blk = ix; blk < num_blocks; blk += it_size) {
        uint block_off = src0_row + blk * Q3K_BSIZE;

        float d = f16_to_f32(load_u16_q3k(src0, block_off + 108));

        // Decode this thread's scale slot
        uint scales_off = block_off + 96;
        uint raw0 = load_u32_q3k(src0, scales_off + 0);
        uint raw4 = load_u32_q3k(src0, scales_off + 4);
        uint raw8 = load_u32_q3k(src0, scales_off + 8);

        uint aux_word;
        if      (sub == 0u) aux_word = (raw0 & kmask2)         | (((raw8 >> 0u) & kmask1) << 4u);
        else if (sub == 1u) aux_word = (raw4 & kmask2)         | (((raw8 >> 2u) & kmask1) << 4u);
        else if (sub == 2u) aux_word = ((raw0 >> 4u) & kmask2) | (((raw8 >> 4u) & kmask1) << 4u);
        else                aux_word = ((raw4 >> 4u) & kmask2) | (((raw8 >> 6u) & kmask1) << 4u);
        int scale_signed = int((aux_word >> pos) & 0xFFu) - 32;
        float scale_d = d * float(scale_signed);

        // 16 qs bytes for this sub-block
        uint qs_block_off = block_off + 32u + qs_off;
        uint qw0 = load_u32_q3k(src0, qs_block_off + 0);
        uint qw1 = load_u32_q3k(src0, qs_block_off + 4);
        uint qw2 = load_u32_q3k(src0, qs_block_off + 8);
        uint qw3 = load_u32_q3k(src0, qs_block_off + 12);

        // 16 hmask bytes
        uint hm_block_off = block_off + 0u + hm_off;
        uint hw0 = load_u32_q3k(src0, hm_block_off + 0);
        uint hw1 = load_u32_q3k(src0, hm_block_off + 4);
        uint hw2 = load_u32_q3k(src0, hm_block_off + 8);
        uint hw3 = load_u32_q3k(src0, hm_block_off + 12);

        uint qp0 = (qw0 >> shift) & 0x03030303u;
        uint qp1 = (qw1 >> shift) & 0x03030303u;
        uint qp2 = (qw2 >> shift) & 0x03030303u;
        uint qp3 = (qw3 >> shift) & 0x03030303u;

        uint hp0 = (hw0 >> m_bit) & 0x01010101u;
        uint hp1 = (hw1 >> m_bit) & 0x01010101u;
        uint hp2 = (hw2 >> m_bit) & 0x01010101u;
        uint hp3 = (hw3 >> m_bit) & 0x01010101u;

        // Activations: 16 floats coalesced via Load4
        // src1 is F32 contiguous, so src1_base + (offset_in_floats * 4) is 4-aligned.
        uint y_off = src1_base + (blk * QK_K + is * 16u) * 4u;
        uint4 y0 = src1.Load4(y_off + 0);
        uint4 y1 = src1.Load4(y_off + 16);
        uint4 y2 = src1.Load4(y_off + 32);
        uint4 y3 = src1.Load4(y_off + 48);

        float by0  = asfloat(y0.x); float by1  = asfloat(y0.y);
        float by2  = asfloat(y0.z); float by3  = asfloat(y0.w);
        float by4  = asfloat(y1.x); float by5  = asfloat(y1.y);
        float by6  = asfloat(y1.z); float by7  = asfloat(y1.w);
        float by8  = asfloat(y2.x); float by9  = asfloat(y2.y);
        float by10 = asfloat(y2.z); float by11 = asfloat(y2.w);
        float by12 = asfloat(y3.x); float by13 = asfloat(y3.y);
        float by14 = asfloat(y3.z); float by15 = asfloat(y3.w);

        float q0  = float((qp0      ) & 0xFFu) + float((hp0      ) & 0xFFu) * 4.0f - 4.0f;
        float q1  = float((qp0 >>  8) & 0xFFu) + float((hp0 >>  8) & 0xFFu) * 4.0f - 4.0f;
        float q2  = float((qp0 >> 16) & 0xFFu) + float((hp0 >> 16) & 0xFFu) * 4.0f - 4.0f;
        float q3  = float((qp0 >> 24) & 0xFFu) + float((hp0 >> 24) & 0xFFu) * 4.0f - 4.0f;
        float q4  = float((qp1      ) & 0xFFu) + float((hp1      ) & 0xFFu) * 4.0f - 4.0f;
        float q5  = float((qp1 >>  8) & 0xFFu) + float((hp1 >>  8) & 0xFFu) * 4.0f - 4.0f;
        float q6  = float((qp1 >> 16) & 0xFFu) + float((hp1 >> 16) & 0xFFu) * 4.0f - 4.0f;
        float q7  = float((qp1 >> 24) & 0xFFu) + float((hp1 >> 24) & 0xFFu) * 4.0f - 4.0f;
        float q8  = float((qp2      ) & 0xFFu) + float((hp2      ) & 0xFFu) * 4.0f - 4.0f;
        float q9  = float((qp2 >>  8) & 0xFFu) + float((hp2 >>  8) & 0xFFu) * 4.0f - 4.0f;
        float q10 = float((qp2 >> 16) & 0xFFu) + float((hp2 >> 16) & 0xFFu) * 4.0f - 4.0f;
        float q11 = float((qp2 >> 24) & 0xFFu) + float((hp2 >> 24) & 0xFFu) * 4.0f - 4.0f;
        float q12 = float((qp3      ) & 0xFFu) + float((hp3      ) & 0xFFu) * 4.0f - 4.0f;
        float q13 = float((qp3 >>  8) & 0xFFu) + float((hp3 >>  8) & 0xFFu) * 4.0f - 4.0f;
        float q14 = float((qp3 >> 16) & 0xFFu) + float((hp3 >> 16) & 0xFFu) * 4.0f - 4.0f;
        float q15 = float((qp3 >> 24) & 0xFFu) + float((hp3 >> 24) & 0xFFu) * 4.0f - 4.0f;

        float sum_qy = q0*by0 + q1*by1 + q2*by2  + q3*by3
                     + q4*by4 + q5*by5 + q6*by6  + q7*by7
                     + q8*by8 + q9*by9 + q10*by10+ q11*by11
                     + q12*by12+q13*by13+q14*by14+q15*by15;

        acc += scale_d * sum_qy;
    }

    // Wave-intrinsic reduction
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WARP_SIZE;
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WARP_SIZE;
    if (num_waves <= WARP_SIZE) {
        if (tid < num_waves) {
            float v = shared_acc[tid];
            v = WaveActiveSum(v);
            if (tid == 0) shared_acc[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
    } else {
        for (uint s = num_waves / 2; s > 0; s /= 2) {
            if (tid < s) shared_acc[tid] += shared_acc[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (tid == 0) {
        float result = shared_acc[0];
        result += load_fused_bias(i0, i2, i3);
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
