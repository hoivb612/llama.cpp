// mul_mat_vec_q3k_mr.hlsl - Multi-row Q3_K matvec (M=1, 2 rows/group)
//
// Q3_K block (110 bytes — NOT 4-byte aligned):
//   offset   0..31  : hmask[32] (1 bit per element, high bit of 3-bit quant)
//   offset  32..95  : qs[64]    (2 bits per element, low bits)
//   offset  96..107 : scales[12] (16 packed 6-bit values, biased; subtract 32)
//   offset 108..109 : d (fp16)
//
// 256 threads, per-element dequant, shares activation loads across 2 rows.
// NB: this shader is only selected when K >= 4096; smaller K falls back to CPU
// (see supports_op for the matvec gating).
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q3K_BSIZE   110

groupshared float shared_acc[64];

uint read_byte_q3k_mr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float dequant_q3k_element(uint src0_row, uint k) {
    uint blk  = k / QK_K;
    uint elem = k % QK_K;
    uint block_off = src0_row + blk * Q3K_BSIZE;

    // d at offsets 108..109; word at 108 covers bytes 108..111
    uint d_word = read_byte_q3k_mr(src0, block_off + 108) |
                 (read_byte_q3k_mr(src0, block_off + 109) << 8);
    float d = f16_to_f32(d_word & 0xFFFFu);

    uint n      = elem >> 7;        // 0 or 1
    uint y_in_n = elem & 127u;
    uint j      = y_in_n >> 5;      // 0..3
    uint pos    = y_in_n & 31u;
    uint l      = pos & 15u;
    uint half   = pos >> 4;
    uint shift  = j << 1;
    uint is     = (n << 3) + (j << 1) + half;        // 0..15
    // hmask is NOT advanced across n; bit index per (n,j)
    uint m_bit  = (n << 2) + j;                       // 0..7
    // qs offset within qs[]: q_base advances 32 per n, then half*16 + l
    uint qs_off = (n << 5) + (half << 4) + l;
    // hmask offset: half*16 + l (l in [0,16), so hm_off in [0,32))
    uint hm_off = (half << 4) + l;

    // Decode this element's scale (signed 6-bit, biased by 32)
    // scales[16] are packed into 12 bytes at offsets 96..107
    uint sub = is >> 2;          // 0..3
    uint pos_byte = is & 3u;     // 0..3
    uint scales_off = block_off + 96;

    // We only need 1 byte of the 4 packed scale words (sc_w0..sc_w3 in batch shader).
    // sub==0: sc_w0 = (raw0 & 0x0F0F0F0F) | (((raw8>>0)&0x03030303)<<4)
    // sub==1: sc_w1 = (raw4 & 0x0F0F0F0F) | (((raw8>>2)&0x03030303)<<4)
    // sub==2: sc_w2 = ((raw0>>4) & 0x0F0F0F0F) | (((raw8>>4)&0x03030303)<<4)
    // sub==3: sc_w3 = ((raw4>>4) & 0x0F0F0F0F) | (((raw8>>6)&0x03030303)<<4)
    uint base_sel = (sub & 1u) ? 4u : 0u;        // raw4 if sub odd, else raw0
    uint base_byte = read_byte_q3k_mr(src0, scales_off + base_sel + pos_byte);
    uint hi_byte   = read_byte_q3k_mr(src0, scales_off + 8u + pos_byte);
    uint base_nib  = (sub < 2u) ? (base_byte & 0x0Fu) : (base_byte >> 4);
    uint hi_shift  = sub << 1;                   // 0,2,4,6
    uint hi_nib    = (hi_byte >> hi_shift) & 0x03u;
    int  scale_signed = int(base_nib | (hi_nib << 4)) - 32;
    float scale_d  = d * float(scale_signed);

    uint qb = read_byte_q3k_mr(src0, block_off + 32u + qs_off);
    uint hb = read_byte_q3k_mr(src0, block_off + 0u + hm_off);
    int q_lo = int((qb >> shift) & 3u);
    int q_hi = ((hb >> m_bit) & 1u) ? 0 : 4;

    return scale_d * float(q_lo - q_hi);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint k = tid; k < K; k += GROUP_SIZE) {
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc0 += dequant_q3k_element(src0_row0, k) * x;
        acc1 += dequant_q3k_element(src0_row1, k) * x;
    }

    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id]      = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid]      += shared_acc[tid + s];
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
