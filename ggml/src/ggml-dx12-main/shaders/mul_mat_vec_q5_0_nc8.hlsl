// Q5_0 small-M batch matmul: one output row per lane, up to 8 columns per group.

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK5_0 32
#define Q5_0_BSIZE 22
#define NUM_COLS 8

uint read_u32_q50_nc8(uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = src0.Load(aligned);
    uint hi = src0.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_q50_nc8(uint byte_off) {
    uint word = src0.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xffffu);
}

int decode_q50_nc8(uint block_off, uint qh, uint elem) {
    uint qs_idx = elem & 15u;
    uint qs_word = read_u32_q50_nc8(block_off + 6u + (qs_idx & ~3u));
    uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xffu;
    if (elem < 16u) {
        uint high = ((qh >> elem) << 4u) & 0x10u;
        return (int)((qs_byte & 0x0fu) | high) - 16;
    }
    uint e = elem - 16u;
    uint high = (qh >> (e + 12u)) & 0x10u;
    return (int)((qs_byte >> 4u) | high) - 16;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row = group_id.x * GROUP_SIZE + local_id;
    if (row >= ne0) {
        return;
    }
    uint col_base = group_id.y * NUM_COLS;
    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;
    uint src0_row = src0_offset + row * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_batch = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc[NUM_COLS];
    [unroll]
    for (uint c = 0; c < NUM_COLS; ++c) {
        acc[c] = 0.0f;
    }

    uint num_blocks = ne00 / QK5_0;
    for (uint block = 0; block < num_blocks; ++block) {
        uint block_off = src0_row + block * Q5_0_BSIZE;
        float d = read_f16_q50_nc8(block_off);
        uint qh = read_u32_q50_nc8(block_off + 2u);
        uint k_base = block * QK5_0;

        [unroll]
        for (uint j = 0; j < 16u; ++j) {
            int q0 = decode_q50_nc8(block_off, qh, j);
            int q1 = decode_q50_nc8(block_off, qh, j + 16u);
            [unroll]
            for (uint c = 0; c < NUM_COLS; ++c) {
                uint col = col_base + c;
                if (col < ne1) {
                    uint col_base_off = src1_batch + col * nb11;
                    float x0 = asfloat(src1.Load(col_base_off + (k_base + j) * nb10));
                    float x1 = asfloat(src1.Load(col_base_off + (k_base + j + 16u) * nb10));
                    acc[c] += d * (float)q0 * x0;
                    acc[c] += d * (float)q1 * x1;
                }
            }
        }
    }

    float bias = load_fused_bias(row, i2, i3);
    [unroll]
    for (uint c = 0; c < NUM_COLS; ++c) {
        uint col = col_base + c;
        if (col < ne1) {
            uint off = offset_4d(row, col, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, acc[c] + bias, dst_esize);
        }
    }
}
