// mul_mat_q4k.hlsl - Matrix multiplication with Q4_K quantized weights
// Dequantizes Q4_K on the fly during dot product.
// Uses the exact same scale decoding as Vulkan's dequant_q4_k.comp.
#include "ggml_common.hlsli"

#define QK_K 256
#define Q4K_BLOCK_SIZE 144  // 2+2+12+128 bytes

// Read a single byte from a ByteAddressBuffer at an arbitrary byte offset
uint read_byte(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

// Dequantize one element from a Q4_K block
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
    uint sc_idx, mb_val;

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

        sc_idx = (read_byte(buf, scales_off + scidx0) & 0x0Fu) |
                 ((read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb_val = ((read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
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

        sc_idx = (read_byte(buf, scales_off + scidx0) & 0x0Fu) |
                 ((read_byte(buf, scales_off + scidx1) & scmask1) >> scshift1);
        mb_val = ((read_byte(buf, scales_off + mbidx0) & mbmask0) >> mbshift0) |
                 ((read_byte(buf, scales_off + mbidx1) & mbmask1) >> mbshift1);
    }

    float d = dall * (float)sc_idx;
    float m = dmin_val * (float)mb_val;

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

// Optimized flat MUL_MAT with Q4_K: process 32 elements at a time
// Amortizes scale/min decoding across each 32-element half sub-block
// instead of decoding per-element.

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

    // Broadcast: clamp batch indices for src0
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    // Base offset for weight row i0 in src0
    uint src0_row_off = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;

    // Pre-compute src1 base for this row
    uint src1_row_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row_off + block * Q4K_BLOCK_SIZE;
        uint k_base = block * QK_K;

        // Load block header once per block
        uint dm_raw = src0.Load(block_off);
        float dall = f16_to_f32(dm_raw & 0xFFFFu);
        float dmin_val = f16_to_f32(dm_raw >> 16);

        uint scales_off = block_off + 4;
        uint qs_off = block_off + 16;

        // Process 8 sub-blocks of 32 elements each (4 pairs of low/high)
        for (uint il = 0; il < 4; il++) {
            uint is = 2 * il;

            // Decode scale/min for low half (is) and high half (is+1)
            uint sc_lo, mb_lo, sc_hi, mb_hi;
            {
                // Low half scale/min
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

                sc_lo = (read_byte(src0, scales_off + scidx0) & 0x0Fu) |
                        ((read_byte(src0, scales_off + scidx1) & scmask1) >> scshift1);
                mb_lo = ((read_byte(src0, scales_off + mbidx0) & mbmask0) >> mbshift0) |
                        ((read_byte(src0, scales_off + mbidx1) & mbmask1) >> mbshift1);

                // High half scale/min (is+1)
                uint is1 = is + 1;
                scidx0 = (is < 4) ? is1 : (is1 + 4);
                scidx1 = (is < 4) ? is1 : (is1 - 4);
                mbidx0 = is1 + 4;
                mbidx1 = (is < 4) ? is1 + 4 : is1;

                sc_hi = (read_byte(src0, scales_off + scidx0) & 0x0Fu) |
                        ((read_byte(src0, scales_off + scidx1) & scmask1) >> scshift1);
                mb_hi = ((read_byte(src0, scales_off + mbidx0) & mbmask0) >> mbshift0) |
                        ((read_byte(src0, scales_off + mbidx1) & mbmask1) >> mbshift1);
            }

            float d_lo = dall * (float)sc_lo;
            float m_lo = dmin_val * (float)mb_lo;
            float d_hi = dall * (float)sc_hi;
            float m_hi = dmin_val * (float)mb_hi;

            // Process 32 low-nibble elements
            uint qs_base = qs_off + il * 32;
            uint k_lo = k_base + il * 64;
            for (uint j = 0; j < 32; j += 4) {
                // Load 4 bytes at once (4 qs values)
                uint qs4 = src0.Load(qs_base + j);
                float q0 = (float)(qs4 & 0x0Fu);
                float q1 = (float)((qs4 >> 8) & 0x0Fu);
                float q2 = (float)((qs4 >> 16) & 0x0Fu);
                float q3 = (float)((qs4 >> 24) & 0x0Fu);

                uint off_base = src1_row_base + (k_lo + j) * nb10;
                acc += (d_lo * q0 - m_lo) * load_auto(src1, off_base, src1_esize);
                acc += (d_lo * q1 - m_lo) * load_auto(src1, off_base + nb10, src1_esize);
                acc += (d_lo * q2 - m_lo) * load_auto(src1, off_base + 2 * nb10, src1_esize);
                acc += (d_lo * q3 - m_lo) * load_auto(src1, off_base + 3 * nb10, src1_esize);
            }

            // Process 32 high-nibble elements
            uint k_hi = k_base + il * 64 + 32;
            for (uint j = 0; j < 32; j += 4) {
                uint qs4 = src0.Load(qs_base + j);
                float q0 = (float)((qs4 >> 4) & 0x0Fu);
                float q1 = (float)((qs4 >> 12) & 0x0Fu);
                float q2 = (float)((qs4 >> 20) & 0x0Fu);
                float q3 = (float)((qs4 >> 28) & 0x0Fu);

                uint off_base = src1_row_base + (k_hi + j) * nb10;
                acc += (d_hi * q0 - m_hi) * load_auto(src1, off_base, src1_esize);
                acc += (d_hi * q1 - m_hi) * load_auto(src1, off_base + nb10, src1_esize);
                acc += (d_hi * q2 - m_hi) * load_auto(src1, off_base + 2 * nb10, src1_esize);
                acc += (d_hi * q3 - m_hi) * load_auto(src1, off_base + 3 * nb10, src1_esize);
            }
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
