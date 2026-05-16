// get_rows_q4_1.hlsl - Gather and dequantize rows from Q4_1 source
#include "ggml_common.hlsli"

#define QK4_1 32
#define Q4_1_BSIZE 20

float read_f16_gr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_byte_gr(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
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

    uint block_idx = i0 / QK4_1;
    uint elem_in_block = i0 % QK4_1;

    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_off = row_off + block_idx * Q4_1_BSIZE;

    float d = read_f16_gr(src0, block_off);
    float m = read_f16_gr(src0, block_off + 2);
    uint j = elem_in_block;
    float val;
    if (j < 16) {
        uint qs_byte = read_byte_gr(src0, block_off + 4 + j);
        val = (float)(qs_byte & 0x0Fu) * d + m;
    } else {
        uint qs_byte = read_byte_gr(src0, block_off + 4 + j - 16);
        val = (float)(qs_byte >> 4) * d + m;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, val, dst_esize);
}
