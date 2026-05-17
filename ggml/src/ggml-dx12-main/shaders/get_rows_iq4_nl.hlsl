// get_rows_iq4_nl.hlsl - Gather and dequantize rows from IQ4_NL source
#include "ggml_common.hlsli"

#define QK4_NL 32
#define IQ4_NL_BSIZE 18

int kvalues_iq4nl_gr(uint idx) {
    static const uint packed[4] = {
        0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
    };
    uint w = packed[idx >> 2];
    uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}

float read_f16_gr_iq(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_byte_gr_iq(ByteAddressBuffer buf, uint byte_off) {
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

    uint block_idx = i0 / QK4_NL;
    uint elem_in_block = i0 % QK4_NL;

    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_off = row_off + block_idx * IQ4_NL_BSIZE;

    float d = read_f16_gr_iq(src0, block_off);

    uint j = elem_in_block;
    int val;
    if (j < 16) {
        uint qs_byte = read_byte_gr_iq(src0, block_off + 2 + j);
        val = kvalues_iq4nl_gr(qs_byte & 0x0Fu);
    } else {
        uint jj = j - 16;
        uint qs_byte = read_byte_gr_iq(src0, block_off + 2 + jj);
        val = kvalues_iq4nl_gr((qs_byte >> 4) & 0x0Fu);
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, d * (float)val, dst_esize);
}
