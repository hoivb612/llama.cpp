// get_rows_q6k.hlsl - Gather and dequantize rows from Q6_K source
#include "ggml_common.hlsli"

#define QK_K 256
#define Q6K_BLOCK_SIZE 210

uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int read_sbyte(ByteAddressBuffer buf, uint byte_off) {
    uint b = read_byte(buf, byte_off);
    return (b < 128) ? (int)b : (int)b - 256;
}

float dequant_q6k_element(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    uint ql_off = block_off;
    uint qh_off = block_off + 128;
    uint scales_off = block_off + 192;
    uint d_off = block_off + 208;

    // d is f16 — block_off may be 2-byte misaligned (Q6K_BLOCK_SIZE=210)
    uint d_word = buf.Load(d_off & ~3u);
    float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    uint ip = elem_in_block / 128;
    uint il = elem_in_block % 128;

    uint is = 8 * ip + il / 16;
    int scale = read_sbyte(buf, scales_off + is);

    uint ql_idx = 64 * ip + (il % 64);
    uint ql_val = read_byte(buf, ql_off + ql_idx);

    uint qh_val = read_byte(buf, qh_off + 32 * ip + (il % 32));

    int q;
    if (il < 32) {
        q = (int)((ql_val & 0x0Fu) | (((qh_val >> 0) & 3u) << 4)) - 32;
    } else if (il < 64) {
        q = (int)((ql_val & 0x0Fu) | (((qh_val >> 2) & 3u) << 4)) - 32;
    } else if (il < 96) {
        q = (int)((ql_val >> 4) | (((qh_val >> 4) & 3u) << 4)) - 32;
    } else {
        q = (int)((ql_val >> 4) | (((qh_val >> 6) & 3u) << 4)) - 32;
    }

    return d * (float)scale * (float)q;
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

    uint block_idx = i0 / QK_K;
    uint elem_in_block = i0 % QK_K;

    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_off = row_off + block_idx * Q6K_BLOCK_SIZE;

    float dequant_val = dequant_q6k_element(src0, block_off, elem_in_block);

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, dequant_val, dst_esize);
}
