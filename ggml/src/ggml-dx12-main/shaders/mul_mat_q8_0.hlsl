// mul_mat_q8_0.hlsl - Matrix multiplication with Q8_0 quantized weights
// Q8_0 block: d(f16) + qs[32](int8) = 34 bytes per 32 elements
// Optimized: packed uint32 loads for 4 int8 weights at a time
#include "ggml_common.hlsli"

#define QK8_0 32
#define Q8_0_BSIZE 34

float read_f16(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint shift = (byte_off & 2u) * 8u;
    return f16_to_f32((word >> shift) & 0xFFFFu);
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

    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i1 * nb11 + i2 * nb12 + i3 * nb13;

    float acc = 0.0f;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_off = src0_row + block * Q8_0_BSIZE;
        float d = read_f16(src0, block_off);
        uint qs_base = block_off + 2;
        uint k_base = block * QK8_0;

        // Process 4 int8 values per iteration (8 iterations per block)
        for (uint j = 0; j < 32; j += 4) {
            uint packed = read_u32_fast(src0, qs_base + j);

            // Unpack 4 int8 values and multiply with input
            [unroll]
            for (uint jj = 0; jj < 4; jj++) {
                uint raw = (packed >> (jj * 8u)) & 0xFFu;
                int q = (raw < 128u) ? (int)raw : (int)raw - 256;
                float x = load_auto(src1, src1_base + (k_base + j + jj) * nb10, src1_esize);
                acc += d * (float)q * x;
            }
        }
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
