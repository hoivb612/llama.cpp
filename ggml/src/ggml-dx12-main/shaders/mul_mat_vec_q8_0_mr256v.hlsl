// mul_mat_vec_q8_0_mr256v.hlsl - Vectorized Q8_0 multi-row matvec (256 threads)
//
// Processes 4 Q8_0 elements per thread per iteration using packed uint32 loads.
// Matches Vulkan's K_PER_ITER=8 strategy (dequantize4 × 2) but adapted to HLSL.
// 2 rows per group, 256 threads, tree reduction.
//
// Q8_0 block: d(f16) + qs[32](int8) = 34 bytes per 32 elements
// Each thread handles 4 consecutive int8 values via a single uint32 load.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch

#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define QK8_0 32
#define Q8_0_BSIZE 34

groupshared float shared_acc[64];

float read_f16_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

// Read 4 packed int8 as a uint32 (misalignment-safe)
uint read_u32_fast(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    if (shift == 0u) return lo;
    uint hi = buf.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

// Unpack 4 signed int8 from uint32 and dot-product with 4 floats
float dot4_q8(uint packed_q, float x0, float x1, float x2, float x3) {
    // Extract 4 signed int8 values
    int q0 = int(packed_q & 0xFFu); if (q0 >= 128) q0 -= 256;
    int q1 = int((packed_q >> 8) & 0xFFu); if (q1 >= 128) q1 -= 256;
    int q2 = int((packed_q >> 16) & 0xFFu); if (q2 >= 128) q2 -= 256;
    int q3 = int(packed_q >> 24); if (q3 >= 128) q3 -= 256;

    return float(q0) * x0 + float(q1) * x1 + float(q2) * x2 + float(q3) * x3;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(group_id) * 2;
    if (row0 >= ne0) return;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_blocks = K / QK8_0;
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    // Each thread processes 4 elements per iteration (stride = GROUP_SIZE * 4)
    // With 256 threads, that's 1024 elements per iteration = 32 Q8_0 blocks
    uint k = tid * 4;
    for (; k + 3 < K; k += GROUP_SIZE * 4) {
        // Shared: load 4 F32 activation values
        uint4 xp = src1.Load4(src1_base + k * 4);
        float x0 = asfloat(xp.x);
        float x1 = asfloat(xp.y);
        float x2 = asfloat(xp.z);
        float x3 = asfloat(xp.w);

        // Block and element indices for the 4 elements
        uint block0 = k / QK8_0;
        uint elem0 = k % QK8_0;

        // All 4 elements may span 1 or 2 blocks (if elem0 >= 29)
        // For simplicity and correctness, handle per-element when crossing block boundary
        if (elem0 + 3 < QK8_0) {
            // Common case: all 4 in same block
            uint boff0 = src0_row0 + block0 * Q8_0_BSIZE;
            float d0 = read_f16_v(src0, boff0);
            uint q0_packed = read_u32_fast(src0, boff0 + 2 + elem0);
            acc0 += d0 * dot4_q8(q0_packed, x0, x1, x2, x3);

            uint boff1 = src0_row1 + block0 * Q8_0_BSIZE;
            float d1 = read_f16_v(src0, boff1);
            uint q1_packed = read_u32_fast(src0, boff1 + 2 + elem0);
            acc1 += d1 * dot4_q8(q1_packed, x0, x1, x2, x3);
        } else {
            // Rare: block boundary crossing — process individually
            [unroll]
            for (uint j = 0; j < 4; j++) {
                uint kk = k + j;
                uint blk = kk / QK8_0;
                uint elm = kk % QK8_0;
                float xv = (j == 0) ? x0 : (j == 1) ? x1 : (j == 2) ? x2 : x3;

                uint bo0 = src0_row0 + blk * Q8_0_BSIZE;
                float dd0 = read_f16_v(src0, bo0);
                uint word0 = src0.Load((bo0 + 2 + elm) & ~3u);
                uint raw0 = (word0 >> (((bo0 + 2 + elm) & 3u) * 8u)) & 0xFFu;
                int qq0 = (raw0 < 128u) ? (int)raw0 : (int)raw0 - 256;
                acc0 += dd0 * float(qq0) * xv;

                uint bo1 = src0_row1 + blk * Q8_0_BSIZE;
                float dd1 = read_f16_v(src0, bo1);
                uint word1 = src0.Load((bo1 + 2 + elm) & ~3u);
                uint raw1 = (word1 >> (((bo1 + 2 + elm) & 3u) * 8u)) & 0xFFu;
                int qq1 = (raw1 < 128u) ? (int)raw1 : (int)raw1 - 256;
                acc1 += dd1 * float(qq1) * xv;
            }
        }
    }
    // Handle remainder (< 4 elements)
    for (; k < K; k++) {
        float x = asfloat(src1.Load(src1_base + k * 4));
        uint block = k / QK8_0;
        uint elem = k % QK8_0;

        uint boff0 = src0_row0 + block * Q8_0_BSIZE;
        float d0 = read_f16_v(src0, boff0);
        uint word0 = src0.Load((boff0 + 2 + elem) & ~3u);
        uint raw0 = (word0 >> (((boff0 + 2 + elem) & 3u) * 8u)) & 0xFFu;
        int q0 = (raw0 < 128u) ? (int)raw0 : (int)raw0 - 256;
        acc0 += d0 * float(q0) * x;

        uint boff1 = src0_row1 + block * Q8_0_BSIZE;
        float d1 = read_f16_v(src0, boff1);
        uint word1 = src0.Load((boff1 + 2 + elem) & ~3u);
        uint raw1 = (word1 >> (((boff1 + 2 + elem) & 3u) * 8u)) & 0xFFu;
        int q1 = (raw1 < 128u) ? (int)raw1 : (int)raw1 - 256;
        acc1 += d1 * float(q1) * x;
    }

    // Two-level reduction with tree reduction
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;
    uint row1_off = (num_waves > 0) ? num_waves : 1;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[row1_off + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[row1_off + tid] += shared_acc[row1_off + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[row1_off];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
