// mul_mat_q8_0_q8_1.hlsl - Batch MUL_MAT using dp4a (dot4add_i8packed)
// Weights: Q8_0 (34 bytes/block: d(f16) + qs[32](int8))
// Input:   Q8_1 (36 bytes/block: ds(2xf16) + qs[32](int8)) in flat scratch buffer
// 
// The Q8_1 scratch buffer is flat: row i1 starts at i1 * num_blocks * 36
// NOT using nb10/nb11 strides (those are for the original F32 tensor)

#include "ggml_common.hlsli"

#define QK8_0 32
#define Q8_0_BSIZE 34
#define Q8_1_BSIZE 36

float read_f16_fast(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    uint K = ne00;
    uint num_blocks = K / QK8_0;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    // Q8_0 weight row
    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;

    // Q8_1 quantized input: flat layout in scratch buffer
    // Row i1 of batch (i2,i3) starts at: ((i3*ne12 + i2)*ne11 + i1) * num_blocks * Q8_1_BSIZE
    uint flat_row = (i3 * ne12 + i2) * ne11 + i1;
    uint src1_row = src1_offset + flat_row * num_blocks * Q8_1_BSIZE;

    precise float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        // Q8_0 weight block
        uint w_off = src0_row + block * Q8_0_BSIZE;
        float w_d = read_f16_fast(src0, w_off);
        uint w_qs = w_off + 2;

        // Q8_1 input block (contiguous in scratch)
        uint i_off = src1_row + block * Q8_1_BSIZE;
        uint ds_word = src1.Load(i_off);
        float i_d = f16_to_f32(ds_word & 0xFFFFu);
        uint i_qs = i_off + 4;

        // 8 dp4a operations per block
        int isum = 0;
        [unroll]
        for (uint j = 0; j < 32; j += 4) {
            uint w_packed = read_u32_fast(src0, w_qs + j);
            uint i_packed = src1.Load(i_qs + j);
            isum = dot4add_i8packed(w_packed, i_packed, isum);
        }

        acc += w_d * i_d * (float)isum;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}