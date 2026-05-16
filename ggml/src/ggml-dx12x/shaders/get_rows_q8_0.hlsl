// get_rows_q8_0.hlsl - Gather and dequantize rows from Q8_0 source
#include "ggml_common.hlsli"

#define QK8_0 32
#define Q8_0_BSIZE 34

float read_f16_gr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

int read_sbyte_gr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint b = (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
    return (b < 128) ? (int)b : (int)b - 256;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint row_idx_off = src1_offset + i1 * nb10 + i2 * nb12 + i3 * nb13;
    int row_idx = asint(src1.Load(row_idx_off));

    uint block_idx = i0 / QK8_0;
    uint elem_in_block = i0 % QK8_0;

    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_off = row_off + block_idx * Q8_0_BSIZE;

    float d = read_f16_gr(src0, block_off);
    int qs = read_sbyte_gr(src0, block_off + 2 + elem_in_block);

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, d * (float)qs, dst_esize);
}
