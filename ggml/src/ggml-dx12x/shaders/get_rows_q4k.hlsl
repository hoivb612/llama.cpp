// get_rows_q4k.hlsl - Gather and dequantize rows from Q4_K source
// src0: Q4_K data table [ne01 rows, ne00 elements per row]
// src1: indices (I32)
// dst:  F32 output
//
// This dequantizes Q4_K blocks on-the-fly while gathering rows.
// Scale decoding matches Vulkan dequant_q4_k.comp exactly.
#include "ggml_common.hlsli"

#define QK_K 256
#define Q4K_BLOCK_SIZE 144

// Read a single byte from a ByteAddressBuffer at an arbitrary byte offset
uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Dequantize one element from a Q4_K block.
// Matches the Vulkan dequant_q4_k.comp scale decoding exactly.
float dequant_q4k_element(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    // Load d and dmin (packed f16 pair)
    uint dm_raw = buf.Load(block_off);
    float dall = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);

    uint scales_off = block_off + 4;

    // Determine sub-block: 8 sub-blocks of 32 elements
    // Vulkan layout: il = sub_pair (0..3), low 32 use d1/m1, high 32 use d2/m2
    uint il = elem_in_block / 64;       // which 64-element chunk (0..3)
    uint elem_in_chunk = elem_in_block % 64;
    bool is_high = (elem_in_chunk >= 32);
    uint elem_in_half = elem_in_chunk % 32;

    uint is = 2 * il; // scale pair index

    // Decode scale and min for the appropriate sub-block
    // This matches the Vulkan dequant_q4_k.comp exactly
    uint sc, mb;

    if (!is_high) {
        // Low nibble sub-block (is)
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

        sc = (read_byte(buf, scales_off + scidx0) & 0x0Fu) |
             ((read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb = ((read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
             ((read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    } else {
        // High nibble sub-block (is+1)
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

        sc = (read_byte(buf, scales_off + scidx0) & 0x0Fu) |
             ((read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb = ((read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
             ((read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    }

    float d = dall * (float)sc;
    float m = dmin_val * (float)mb;

    // Get the quantized 4-bit value
    uint qs_off = block_off + 16; // scales_off + 12
    uint qs_byte_idx = il * 32 + elem_in_half;
    uint qs_byte = read_byte(buf, qs_off + qs_byte_idx);

    float q;
    if (!is_high) {
        q = (float)(qs_byte & 0x0Fu);
    } else {
        q = (float)(qs_byte >> 4);
    }

    return d * q - m;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    // Get source row index from src1
    uint row_idx_off = src1_offset + i1 * nb10 + i2 * nb12 + i3 * nb13;
    int row_idx = asint(src1.Load(row_idx_off));

    // Find which Q4_K block and element within the block
    uint block_idx = i0 / QK_K;
    uint elem_in_block = i0 % QK_K;

    // Block offset in src0
    uint row_off = src0_offset + (uint)row_idx * nb01 + i2 * nb02 + i3 * nb03;
    uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;

    float dequant_val = dequant_q4k_element(src0, block_off, elem_in_block);

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, dequant_val, dst_esize);
}
