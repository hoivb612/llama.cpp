// mul_mat_vec_glu_q3_k.hlsl - Fused MUL_MAT(M=1, W_gate Q3_K) + MUL_MAT(M=1, W_up Q3_K) + GLU(SwiGLU split)
//
// R9 fusion variant for Q3_K weights.
//
// Q3_K block (110 bytes per 256 elements, NOT 4-byte aligned):
//   offset   0..31  : hmask[32] (1 bit per element, high bit of 3-bit quant)
//   offset  32..95  : qs[64]    (2 bits per element, low bits)
//   offset  96..107 : scales[12] (16 packed 6-bit values, biased by 32)
//   offset 108..109 : d (fp16)
//
// Per-element dequant (mirrors mul_mat_vec_q3k_mr.hlsl), 4 roles per element:
// gate row0, gate row1, up row0, up row1. Activation is loaded once and shared
// across all 4 roles. After per-thread accumulation, a wave + LDS reduction
// combines partial sums; thread 0 then applies SwiGLU(gate) * up and writes
// 2 output values.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch * ne2 * ne3
// Gate matrix bound to src0, up matrix bound to src2 (R9 fusion convention).
// Activation (src1) is F32 contiguous.

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q3K_BSIZE   110

groupshared float shared_acc[128]; // 4 roles * max 32 waves (wave_size=8 worst case)

uint read_byte_q3k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float dequant_q3k_element(ByteAddressBuffer buf, uint src0_row, uint k) {
    uint blk  = k / QK_K;
    uint elem = k % QK_K;
    uint block_off = src0_row + blk * Q3K_BSIZE;

    uint d_word = read_byte_q3k(buf, block_off + 108) |
                 (read_byte_q3k(buf, block_off + 109) << 8);
    float d = f16_to_f32(d_word & 0xFFFFu);

    uint n      = elem >> 7;
    uint y_in_n = elem & 127u;
    uint j      = y_in_n >> 5;
    uint pos    = y_in_n & 31u;
    uint l      = pos & 15u;
    uint half   = pos >> 4;
    uint shift  = j << 1;
    uint is     = (n << 3) + (j << 1) + half;
    uint m_bit  = (n << 2) + j;
    uint qs_off = (n << 5) + (half << 4) + l;
    uint hm_off = (half << 4) + l;

    uint sub = is >> 2;
    uint pos_byte = is & 3u;
    uint scales_off = block_off + 96;

    uint base_sel = (sub & 1u) ? 4u : 0u;
    uint base_byte = read_byte_q3k(buf, scales_off + base_sel + pos_byte);
    uint hi_byte   = read_byte_q3k(buf, scales_off + 8u + pos_byte);
    uint base_nib  = (sub < 2u) ? (base_byte & 0x0Fu) : (base_byte >> 4);
    uint hi_shift  = sub << 1;
    uint hi_nib    = (hi_byte >> hi_shift) & 0x03u;
    int  scale_signed = int(base_nib | (hi_nib << 4)) - 32;
    float scale_d  = d * float(scale_signed);

    uint qb = read_byte_q3k(buf, block_off + 32u + qs_off);
    uint hb = read_byte_q3k(buf, block_off + 0u  + hm_off);
    int q_lo = int((qb >> shift) & 3u);
    int q_hi = ((hb >> m_bit) & 1u) ? 0 : 4;

    return scale_d * float(q_lo - q_hi);
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

    uint K = ne00;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint up_base   = op1          + i2_src0 * nb02 + i3_src0 * nb03;
    uint gate_row0 = src0_base + row0 * nb01;
    uint gate_row1 = src0_base + row1 * nb01;
    uint up_row0   = up_base   + row0 * nb01;
    uint up_row1   = up_base   + row1 * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc_g0 = 0.0f;
    precise float acc_g1 = 0.0f;
    precise float acc_u0 = 0.0f;
    precise float acc_u1 = 0.0f;

    for (uint k = tid; k < K; k += GROUP_SIZE) {
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc_g0 += dequant_q3k_element(src0, gate_row0, k) * x;
        acc_g1 += dequant_q3k_element(src0, gate_row1, k) * x;
        acc_u0 += dequant_q3k_element(src2, up_row0,   k) * x;
        acc_u1 += dequant_q3k_element(src2, up_row1,   k) * x;
    }

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
