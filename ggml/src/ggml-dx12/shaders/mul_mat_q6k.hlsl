// mul_mat_q6k.hlsl - Matrix multiplication with Q6_K quantized weights
// Dequantizes Q6_K on the fly during dot product.
// Block layout: ql[128] + qh[64] + scales[16] + d(f16) = 210 bytes
#include "ggml_common.hlsli"

#define QK_K 256
#define Q6K_BLOCK_SIZE 210  // 128+64+16+2 bytes

uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int read_sbyte(ByteAddressBuffer buf, uint byte_off) {
    uint b = read_byte(buf, byte_off);
    return (b < 128) ? (int)b : (int)b - 256;
}

float dequant_q6k_element(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    uint ql_off = block_off;           // ql[128] at offset 0
    uint qh_off = block_off + 128;     // qh[64] at offset 128
    uint scales_off = block_off + 192; // scales[16] at offset 192
    uint d_off = block_off + 208;      // d(f16) at offset 208

    // d is f16 — block_off may be 2-byte misaligned (Q6K_BLOCK_SIZE=210, not multiple of 4)
    uint d_word = buf.Load(d_off & ~3u);
    float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    // Q6_K: 2 halves of 128, each half has 2 sub-blocks of 32 + 2 more of 32
    // Layout matches Vulkan: ip = elem / 128, il = elem % 128
    uint ip = elem_in_block / 128;  // 0 or 1
    uint il = elem_in_block % 128;

    // Scale index: 8*ip + il/16
    uint is = 8 * ip + il / 16;
    int scale = read_sbyte(buf, scales_off + is);

    // ql index: 64*ip + (il % 64)
    uint ql_idx = 64 * ip + (il % 64);
    uint ql_val = read_byte(buf, ql_off + ql_idx);

    // qh: 32*ip + (il % 32)
    uint qh_val = read_byte(buf, qh_off + 32 * ip + (il % 32));

    // Reconstruct 6-bit value depending on which 32-element sub-block
    int q;
    if (il < 32) {
        // First 32: low nibble of ql, bits 0-1 of qh
        q = (int)((ql_val & 0x0Fu) | (((qh_val >> 0) & 3u) << 4)) - 32;
    } else if (il < 64) {
        // Second 32: low nibble of ql+32, bits 2-3 of qh
        q = (int)((ql_val & 0x0Fu) | (((qh_val >> 2) & 3u) << 4)) - 32;
    } else if (il < 96) {
        // Third 32: high nibble of ql (same byte as first 32), bits 4-5 of qh
        q = (int)((ql_val >> 4) | (((qh_val >> 4) & 3u) << 4)) - 32;
    } else {
        // Fourth 32: high nibble of ql+32 (same byte as second 32), bits 6-7 of qh
        q = (int)((ql_val >> 4) | (((qh_val >> 6) & 3u) << 4)) - 32;
    }

    return d * (float)scale * (float)q;
}

// Optimized flat MUL_MAT with Q6_K  
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint K = ne00;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row_off = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    float acc = 0.0f;

    uint num_blocks = K / QK_K;
    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row_off + block * Q6K_BLOCK_SIZE;
        uint k_base = block * QK_K;

        for (uint j = 0; j < QK_K; j++) {
            float w = dequant_q6k_element(src0, block_off, j);
            acc += w * load_auto(src1, src1_base + (k_base + j) * nb10, src1_esize);
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
