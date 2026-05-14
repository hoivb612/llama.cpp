// mul_mat_vec_q8_0_mr256.hlsl - Multi-row Q8_0 matvec, 256 threads (M=1, 2 rows/group)
//
// Q8_0 block: d(f16) + qs[32](int8) = 34 bytes per 32 elements
// 256 threads, shares activation loads across 2 rows.
// Better than 32-thread variant on AMD wave64 (4 full waves vs half a wave).
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

int read_sbyte_v(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    uint b = (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
    return (b < 128) ? (int)b : (int)b - 256;
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
    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    for (uint k = tid; k < K; k += GROUP_SIZE) {
        // Shared: load activation once
        float x = asfloat(src1.Load(src1_base + k * 4));

        uint block = k / QK8_0;
        uint elem = k % QK8_0;

        // Row 0
        uint boff0 = src0_row0 + block * Q8_0_BSIZE;
        float d0 = read_f16_v(src0, boff0);
        int q0 = read_sbyte_v(src0, boff0 + 2 + elem);
        acc0 += d0 * float(q0) * x;

        // Row 1
        uint boff1 = src0_row1 + block * Q8_0_BSIZE;
        float d1 = read_f16_v(src0, boff1);
        int q1 = read_sbyte_v(src0, boff1 + 2 + elem);
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
