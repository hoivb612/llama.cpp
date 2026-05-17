// mul_mat_q5_0.hlsl - Matrix multiplication with Q5_0 quantized weights
// Q5_0 block: d(f16) + qh[4](uint8) + qs[16](uint8) = 22 bytes per 32 elements
// 5-bit signed values: low nibble from qs + high bit from qh, centered at 16
#include "ggml_common.hlsli"

#define QK5_0 32
#define Q5_0_BSIZE 22  // 2 + 4 + 16

float read_f16_q5(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint shift = (byte_off & 2u) * 8u;
    return f16_to_f32((word >> shift) & 0xFFFFu);
}

uint read_byte_q5(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

uint read_u32_q5(ByteAddressBuffer buf, uint byte_off) {
    uint b0 = read_byte_q5(buf, byte_off);
    uint b1 = read_byte_q5(buf, byte_off + 1);
    uint b2 = read_byte_q5(buf, byte_off + 2);
    uint b3 = read_byte_q5(buf, byte_off + 3);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint K = ne00;
    uint num_blocks = K / QK5_0;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * Q5_0_BSIZE;

        float d = read_f16_q5(src0, block_off);
        uint qh = read_u32_q5(src0, block_off + 2);
        uint qs_off = block_off + 6;
        uint k_base = block * QK5_0;

        for (uint j = 0; j < 16; j++) {
            uint qs_byte = read_byte_q5(src0, qs_off + j);

            uint xh_0 = ((qh >> j) << 4) & 0x10u;
            int x0 = (int)((qs_byte & 0x0Fu) | xh_0) - 16;

            uint xh_1 = ((qh >> (j + 12)) & 0x10u);
            int x1 = (int)((qs_byte >> 4) | xh_1) - 16;

            acc += d * (float)x0 * load_auto(src1, src1_base + (k_base + j) * nb10, src1_esize);
            acc += d * (float)x1 * load_auto(src1, src1_base + (k_base + j + 16) * nb10, src1_esize);
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
