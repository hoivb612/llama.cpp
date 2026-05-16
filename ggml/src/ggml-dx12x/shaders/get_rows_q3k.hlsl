// get_rows_q3k.hlsl - Gather and dequantize rows from Q3_K source
// src0: Q3_K data table [ne01 rows, ne00 elements per row]
// src1: indices (I32)
// dst:  F32 output
//
// Q3_K block layout (110 bytes per 256 elements):
//   hmask[32]: high bit (1 per element, bit-packed)
//   qs[64]:    low 2 bits (4 per byte)
//   scales[12]: 6-bit scales, packed
//   d (f16):   super-block scale at offset 108
#include "ggml_common.hlsli"

#define QK_K 256
#define Q3K_BLOCK_SIZE 110

uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Misalignment-safe uint32 load (block size 110 not 4-byte aligned)
uint load_u32_safe(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float dequant_q3k_element(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    // Load d (f16 at offset 108, may be misaligned)
    uint d_off = block_off + 108;
    uint d_word = load_u32_safe(buf, d_off);
    float d_all = f16_to_f32(d_word & 0xFFFFu);

    // Decode position within block
    uint half_idx = elem_in_block / 128;
    uint in_half = elem_in_block % 128;
    uint shift_group = in_half / 32;
    uint sub_group = (in_half % 32) / 16;
    uint pos = in_half % 16;

    // Scale index and 6-bit decode
    uint scale_idx = 8 * half_idx + 2 * shift_group + sub_group;
    uint sc_half = scale_idx / 8;      // 0 or 1
    uint sc_in = scale_idx % 8;         // 0..7
    uint sc_v_im4 = sc_half * 4;
    uint sc_s_shift = sc_v_im4 + 2 * (sc_in / 4);

    uint sc_base = block_off + 96;
    uint raw_low = read_byte(buf, sc_base + sc_in);
    uint raw_high = read_byte(buf, sc_base + (sc_in % 4) + 8);
    uint scale_6bit = ((raw_low >> sc_v_im4) & 0xFu) | (((raw_high >> sc_s_shift) & 3u) << 4);
    float dl = d_all * (float(int(scale_6bit)) - 32.0f);

    // qs value (2 low bits) at offset 32
    uint qs_idx = 32 * half_idx + 16 * sub_group + pos;
    uint shift = shift_group * 2;
    uint q_2bit = (read_byte(buf, block_off + 32 + qs_idx) >> shift) & 3u;

    // hmask value (high bit) at offset 0
    uint hm_idx = 16 * sub_group + pos;
    uint hm_bit = 4 * half_idx + shift_group;
    uint hm_byte = read_byte(buf, block_off + hm_idx);
    uint h = ((hm_byte >> hm_bit) & 1u) != 0u ? 0u : 4u;

    int q_3bit = (int)q_2bit - (int)h;

    return dl * float(q_3bit);
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
    uint block_off = row_off + block_idx * Q3K_BLOCK_SIZE;

    float dequant_val = dequant_q3k_element(src0, block_off, elem_in_block);

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, dequant_val, dst_esize);
}
