// mul_mat_q8_1.hlsl - Matrix multiplication with Q8_1 quantized weights
// Q8_1 block: d(f16) + s(f16) + qs[32](int8) = 36 bytes per 32 elements
// val = qs[j] * d  (s is the sum of quants, not used in dequant)
#include "ggml_common.hlsli"

#define QK8_1 32
#define Q8_1_BSIZE 36

float read_f16_q81(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

int read_sbyte_q81(ByteAddressBuffer buf, uint byte_off) {
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

    uint K = ne00;
    uint num_blocks = K / QK8_1;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * Q8_1_BSIZE;
        float d = read_f16_q81(src0, block_off);
        // s at block_off+2 is sum of quants, not needed for dequant
        uint qs_off = block_off + 4;
        uint k_base = block * QK8_1;

        for (uint j = 0; j < QK8_1; j++) {
            int q = read_sbyte_q81(src0, qs_off + j);
            acc += d * (float)q * load_auto(src1, src1_base + (k_base + j) * nb10, src1_esize);
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
