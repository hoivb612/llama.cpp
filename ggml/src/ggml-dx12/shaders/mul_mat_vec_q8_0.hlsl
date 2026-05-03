// mul_mat_vec_q8_0.hlsl - Specialized matrix-vector multiply for Q8_0 weights (M=1)
//
// Q8_0 block: d(f16) + qs[32](int8) = 34 bytes per 32 elements
// 32 threads cooperate via WaveActiveSum
//
// Dispatch: groups_x = N (output rows), groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define QK8_0 32
#define Q8_0_BSIZE 34

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

int read_sbyte_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint b = (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
    return (b < 128) ? (int)b : (int)b - 256;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint i0 = group_id.y * 65535u + group_id.x;  // linearized 2D for large N (>65535)
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK8_0;

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    // Each thread handles one element per block, iterating over blocks
    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * Q8_0_BSIZE;
        float d = read_f16_v(src0, block_off);
        int q = read_sbyte_v(src0, block_off + 2 + local_id);
        float w = d * (float)q;

        uint k = block * QK8_0 + local_id;
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc += w * x;
    }

    // Wave reduction
    acc = WaveActiveSum(acc);

    if (local_id == 0) {
        if (op0 == 1u) acc += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, acc, dst_esize);
    }
}
