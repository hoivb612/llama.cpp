// get_rows_q3k.hlsl - Dequantize Q3_K rows
// Q3_K block (110 bytes per 256 elements): hmask[32] + qs[64] + scales[12] + d(f16)
#include "ggml_common.hlsli"

#define QK_K 256
#define Q3K_BSIZE 110

uint read_byte_q3k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q3k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    // src1 (indices) is [ne10, ne11, ne12]; strides are nb10/nb11/nb12.
    uint row_idx_off = src1_offset + i1 * nb10 + i2 * nb11 + i3 * nb12;
    int row_idx = asint(src1.Load(row_idx_off));

    uint src0_row = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_idx = i0 / QK_K;
    uint j = i0 % QK_K;
    uint block_off = src0_row + block_idx * Q3K_BSIZE;

    float d = read_f16_q3k(src0, block_off + 108);

    // Canonical Q3_K element layout (matches dequantize_row_q3_K)
    uint sc_idx  = j / 16;                       // sub-block 0..15
    uint shift   = ((sc_idx >> 1) & 3u) * 2u;    // 0,2,4,6
    uint qs_pos  = ((sc_idx >> 3) & 1u) * 32u + (sc_idx & 1u) * 16u + (j & 0xFu);
    uint hm_pos  = (sc_idx & 1u) * 16u + (j & 0xFu);
    uint m_bit   = sc_idx >> 1;                  // 0..7

    uint qs_byte = read_byte_q3k(src0, block_off + 32u + qs_pos);
    uint ql      = (qs_byte >> shift) & 3u;
    uint hm_byte = read_byte_q3k(src0, block_off + 0u + hm_pos);
    uint qh      = (hm_byte >> m_bit) & 1u;
    int q3 = (int)ql - (qh != 0u ? 0 : 4);       // q_low - (h ? 0 : 4)

    // Decode this element's 6-bit scale (matches C scale unpack via aux[0..3])
    uint scales_off = block_off + 96u;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0F0F0F0Fu;
    uint raw0 = read_byte_q3k(src0, scales_off + 0u)
              | (read_byte_q3k(src0, scales_off + 1u) << 8u)
              | (read_byte_q3k(src0, scales_off + 2u) << 16u)
              | (read_byte_q3k(src0, scales_off + 3u) << 24u);
    uint raw4 = read_byte_q3k(src0, scales_off + 4u)
              | (read_byte_q3k(src0, scales_off + 5u) << 8u)
              | (read_byte_q3k(src0, scales_off + 6u) << 16u)
              | (read_byte_q3k(src0, scales_off + 7u) << 24u);
    uint raw8 = read_byte_q3k(src0, scales_off + 8u)
              | (read_byte_q3k(src0, scales_off + 9u) << 8u)
              | (read_byte_q3k(src0, scales_off + 10u) << 16u)
              | (read_byte_q3k(src0, scales_off + 11u) << 24u);
    uint sub = sc_idx >> 2;            // 0..3
    uint pos = (sc_idx & 3u) * 8u;
    uint aux_word;
    if      (sub == 0u) aux_word = (raw0 & kmask2)         | (((raw8 >> 0u) & kmask1) << 4u);
    else if (sub == 1u) aux_word = (raw4 & kmask2)         | (((raw8 >> 2u) & kmask1) << 4u);
    else if (sub == 2u) aux_word = ((raw0 >> 4u) & kmask2) | (((raw8 >> 4u) & kmask1) << 4u);
    else                aux_word = ((raw4 >> 4u) & kmask2) | (((raw8 >> 6u) & kmask1) << 4u);
    int scale = int((aux_word >> pos) & 0xFFu) - 32;

    float val = d * (float)scale * (float)q3;

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, val, dst_esize);
}
