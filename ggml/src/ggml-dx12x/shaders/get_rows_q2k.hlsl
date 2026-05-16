// get_rows_q2k.hlsl - Gather and dequantize rows from Q2_K source
// src0: Q2_K data table [ne01 rows, ne00 elements per row]
// src1: indices (I32)
// dst:  F32 output
//
// Q2_K block layout (84 bytes per 256 elements):
//   scales[16]: 4-bit scale (low) + 4-bit min (high) per group of 16 elements
//   qs[64]:     2-bit quantized values, 4 per byte
//   d (f16):    super-block scale at offset 80
//   dmin (f16): super-block min at offset 82
#include "ggml_common.hlsli"

#define QK_K 256
#define Q2K_BLOCK_SIZE 84

uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

float dequant_q2k_element(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    // Load d and dmin (f16 pair at offset 80)
    uint dm_raw = buf.Load(block_off + 80);
    float d = f16_to_f32(dm_raw & 0xFFFFu);
    float dmin_val = f16_to_f32(dm_raw >> 16);

    // Decode position within block
    // 256 elements: 2 halves of 128, each half has 4 shift groups of 32 (2 sub-groups of 16)
    uint half_idx = elem_in_block / 128;       // 0 or 1
    uint in_half = elem_in_block % 128;
    uint shift_group = in_half / 32;           // 0..3
    uint sub_group = (in_half % 32) / 16;     // 0 or 1
    uint pos = in_half % 16;                   // 0..15

    // Scale index: sequential through all 16 scale bytes
    uint scale_idx = 8 * half_idx + 2 * shift_group + sub_group;
    uint sc_byte = read_byte(buf, block_off + scale_idx);
    float dl = d * float(sc_byte & 0xFu);
    float ml = dmin_val * float(sc_byte >> 4);

    // qs byte and bit shift
    uint qs_idx = 32 * half_idx + 16 * sub_group + pos;
    uint shift = shift_group * 2;
    uint q = (read_byte(buf, block_off + 16 + qs_idx) >> shift) & 3u;

    return dl * float(q) - ml;
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
    uint block_off = row_off + block_idx * Q2K_BLOCK_SIZE;

    float dequant_val = dequant_q2k_element(src0, block_off, elem_in_block);

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, dequant_val, dst_esize);
}
