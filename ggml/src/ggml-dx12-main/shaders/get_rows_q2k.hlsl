// get_rows_q2k.hlsl - Dequantize Q2_K rows
// Q2_K block (84 bytes per 256 elements): scales[16] + qs[64] + d(f16) + dmin(f16)
#include "ggml_common.hlsli"

#define QK_K 256
#define Q2K_BSIZE 84

uint read_byte_q2k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float read_f16_q2k(ByteAddressBuffer buf, uint byte_off) {
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
    uint elem = i0 % QK_K;
    uint block_off = src0_row + block_idx * Q2K_BSIZE;

    float d    = read_f16_q2k(src0, block_off + 80);
    float dmin = read_f16_q2k(src0, block_off + 82);

    uint sc_idx = elem / 16;                      // sub-block 0..15
    uint sc_byte = read_byte_q2k(src0, block_off + sc_idx);
    float scale = d * (float)(sc_byte & 0x0Fu);
    float min_val = dmin * (float)(sc_byte >> 4);

    // Canonical Q2_K element layout (matches dequantize_row_q2_K)
    uint qs_pos = ((sc_idx >> 3) & 1u) * 32u + (sc_idx & 1u) * 16u + (elem & 0xFu);
    uint qs_shift = ((sc_idx >> 1) & 3u) * 2u;    // 0,2,4,6
    uint qs_byte = read_byte_q2k(src0, block_off + 16u + qs_pos);
    int q = (int)((qs_byte >> qs_shift) & 3u);

    float val = scale * (float)q - min_val;

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, val, dst_esize);
}
