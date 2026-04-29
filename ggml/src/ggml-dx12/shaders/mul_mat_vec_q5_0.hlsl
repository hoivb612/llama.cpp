// mul_mat_vec_q5_0.hlsl - Specialized matrix-vector multiply for Q5_0 weights (M=1)
//
// Q5_0 block: d(f16) + qh[4] + qs[16] = 22 bytes per 32 elements
// 32 threads cooperate via WaveActiveSum
// Optimized: vectorized uint32 loads for qh and qs (batch 4 elements per load)
//
// Dispatch: groups_x = N (output rows), groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK5_0 32
#define Q5_0_BSIZE 22

// Fast misalignment-safe uint32 load (2 loads max vs 4 byte reads)
uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint i0 = group_id.x;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK5_0;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    uint elem = local_id;  // element within block (0..31)

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * Q5_0_BSIZE;

        // Load d (f16) and qh (uint32) — vectorized
        float d = read_f16_v(src0, block_off);
        uint qh = read_u32_fast(src0, block_off + 2);

        // Load qs byte — vectorized: 4 threads share one uint32 load
        // qs starts at block_off + 6, each byte holds 2 nibbles
        uint qs_idx = (elem < 16) ? elem : (elem - 16);
        uint qs_word = read_u32_fast(src0, block_off + 6 + (qs_idx & ~3u));
        uint qs_byte = (qs_word >> ((qs_idx & 3u) * 8u)) & 0xFFu;

        int val;
        if (elem < 16) {
            uint xh = ((qh >> elem) << 4) & 0x10u;
            val = (int)((qs_byte & 0x0Fu) | xh) - 16;
        } else {
            uint jj = elem - 16;
            uint xh = ((qh >> (jj + 12)) & 0x10u);
            val = (int)((qs_byte >> 4) | xh) - 16;
        }

        float w = d * (float)val;
        uint k = block * QK5_0 + elem;
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc += w * x;
    }

    acc = WaveActiveSum(acc);

    if (local_id == 0) {
        if (op0 == 1u) acc += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, acc, dst_esize);
    }
}
