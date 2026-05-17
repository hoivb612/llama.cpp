// mul_mat_q5k.hlsl - Matrix multiplication with Q5_K quantized weights
// Dequantizes Q5_K on the fly during dot product.
// Q5_K is like Q4_K but with an extra qh array for the 5th bit.
// Block layout: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] = 176 bytes
#include "ggml_common.hlsli"

#define QK_K 256
#define Q5K_BLOCK_SIZE 176  // 2+2+12+32+128 bytes

uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float dequant_q5k_element(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    // Load d and dmin (packed f16 pair)
    uint dm_raw = buf.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);

    uint scales_off = block_off + 4;
    uint qh_off = block_off + 16;   // 4 + 12
    uint qs_off = block_off + 48;   // 4 + 12 + 32

    // Determine sub-block: 4 chunks of 64, each split into low 32 and high 32
    uint il = elem_in_block / 64;
    uint elem_in_chunk = elem_in_block % 64;
    bool is_high = (elem_in_chunk >= 32);
    uint elem_in_half = elem_in_chunk % 32;

    uint is = 2 * il;

    // Decode scale and min — same as Q4_K
    uint sc_idx, mb_val;

    if (!is_high) {
        uint scidx0 = (is < 4) ? is : (is + 4);
        uint scidx1 = (is < 4) ? is : (is - 4);
        uint scmask1 = (is < 4) ? 0x30u : 0xC0u;
        uint scshift1 = (is < 4) ? 0u : 2u;
        uint mbidx0 = is + 4;
        uint mbidx1 = (is < 4) ? is + 4 : is;
        uint mbmask0 = (is < 4) ? 0x0Fu : 0xF0u;
        uint mbshift0 = (is < 4) ? 0u : 4u;
        uint mbmask1 = (is < 4) ? 0x30u : 0xC0u;
        uint mbshift1 = (is < 4) ? 0u : 2u;

        sc_idx = (read_byte(buf, scales_off + scidx0) & 0x0Fu) |
                 ((read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb_val = ((read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
                 ((read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    } else {
        uint is1 = is + 1;
        uint scidx0 = (is < 4) ? is1 : (is1 + 4);
        uint scidx1 = (is < 4) ? is1 : (is1 - 4);
        uint scmask1 = (is < 4) ? 0x30u : 0xC0u;
        uint scshift1 = (is < 4) ? 0u : 2u;
        uint mbidx0 = is1 + 4;
        uint mbidx1 = (is < 4) ? is1 + 4 : is1;
        uint mbmask0 = (is < 4) ? 0x0Fu : 0xF0u;
        uint mbshift0 = (is < 4) ? 0u : 4u;
        uint mbmask1 = (is < 4) ? 0x30u : 0xC0u;
        uint mbshift1 = (is < 4) ? 0u : 2u;

        sc_idx = (read_byte(buf, scales_off + scidx0) & 0x0Fu) |
                 ((read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb_val = ((read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
                 ((read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    }

    float d = dall * (float)sc_idx;
    float m = dmin_val * (float)mb_val;

    // Get the quantized 4-bit value from qs
    uint qs_byte_idx = il * 32 + elem_in_half;
    uint qs_byte = read_byte(buf, qs_off + qs_byte_idx);

    // Get the high bit from qh
    uint qh_byte = read_byte(buf, qh_off + elem_in_half);
    // hm1 = 1 << (2*il), hm2 = 1 << (2*il+1)
    uint hm = is_high ? (1u << (2u * il + 1u)) : (1u << (2u * il));

    float q;
    if (!is_high) {
        q = (float)((qs_byte & 0x0Fu) + (((qh_byte & hm) != 0u) ? 16u : 0u));
    } else {
        q = (float)((qs_byte >> 4) + (((qh_byte & hm) != 0u) ? 16u : 0u));
    }

    return d * q - m;
}

// Optimized flat MUL_MAT with Q5_K
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint K = ne00;
    uint num_blocks = K / QK_K;

    float acc = 0.0f;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row_off = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row_off + block * Q5K_BLOCK_SIZE;
        uint k_base = block * QK_K;

        for (uint j = 0; j < QK_K; j++) {
            float w = dequant_q5k_element(src0, block_off, j);
            uint k = k_base + j;
            uint off1 = offset_4d(k, i1, i2, i3, nb10, nb11, nb12, nb13, src1_offset);
            acc += w * load_auto(src1, off1, src1_esize);
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
