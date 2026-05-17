// mul_mat_q4_1.hlsl - Matrix multiplication with Q4_1 quantized weights
// Q4_1 block: d(f16) + m(f16) + qs[16](uint8) = 20 bytes per 32 elements
// val = nibble * d + m
#include "ggml_common.hlsli"

#define QK4_1 32
#define Q4_1_BSIZE 20

float read_f16_q41(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_byte_q41(ByteAddressBuffer buf, uint byte_off) {
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

    uint K = ne00;
    uint num_blocks = K / QK4_1;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * Q4_1_BSIZE;
        float d = read_f16_q41(src0, block_off);
        float m = read_f16_q41(src0, block_off + 2);
        uint qs_off = block_off + 4;
        uint k_base = block * QK4_1;

        for (uint j = 0; j < 16; j++) {
            uint qs_byte = read_byte_q41(src0, qs_off + j);
            float x0 = (float)(qs_byte & 0x0Fu) * d + m;
            float x1 = (float)(qs_byte >> 4) * d + m;

            acc += x0 * load_auto(src1, src1_base + (k_base + j) * nb10, src1_esize);
            acc += x1 * load_auto(src1, src1_base + (k_base + j + 16) * nb10, src1_esize);
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
