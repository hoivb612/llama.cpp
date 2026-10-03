// get_rows_mxfp4.hlsl - Gather and dequantize rows from MXFP4 source
#include "ggml_common.hlsli"

#define QK_MXFP4 32
#define MXFP4_BSIZE 17

// kvalues_fp4 holds 2x the E2M1 values, so the E8M0 scale folds in a 0.5.
int kvalues_fp4_gr(uint idx) {
    static const uint packed[4] = {
        0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}

float e8m0_half_gr(uint e) {
    return asfloat((e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23));
}

uint read_byte_gr_mx(ByteAddressBuffer buf, uint byte_off) {
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

    uint row_idx_off = src1_offset + i1 * nb10 + i2 * nb11 + i3 * nb12;
    int row_idx = asint(src1.Load(row_idx_off));

    uint block_idx = i0 / QK_MXFP4;
    uint elem_in_block = i0 % QK_MXFP4;

    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_off = row_off + block_idx * MXFP4_BSIZE;

    float d = e8m0_half_gr(read_byte_gr_mx(src0, block_off));

    uint qs_byte = read_byte_gr_mx(src0, block_off + 1 + (elem_in_block % 16));
    uint nib = (elem_in_block < 16) ? (qs_byte & 0x0Fu) : ((qs_byte >> 4) & 0x0Fu);

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, d * (float)kvalues_fp4_gr(nib), dst_esize);
}
