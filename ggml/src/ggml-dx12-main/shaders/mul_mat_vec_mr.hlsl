// mul_mat_vec_mr.hlsl - Multi-row F16/F32 matvec (M=1)
//
// Processes 2 output rows per workgroup, sharing activation loads.
// Uses Load4 for vectorized F16 weight reads and F32 activation reads.
// Supports F16 (src0_esize=2) and F32 (src0_esize=4) weights.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define NUM_ROWS   2

groupshared float shared_acc[64];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint row0 = group_x_2d(gid) * NUM_ROWS;
    if (row0 >= ne0) return;

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;

    uint src0_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint src0_row0 = src0_base + row0 * nb01;
    uint src0_row1 = src0_base + (row0 + 1) * nb01;

    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc0 = 0.0f;
    precise float acc1 = 0.0f;

    if (src0_esize == 2) {
        // F16 weights: Load4 = 2×uint32 = 4 F16 values per iteration
        // Use mad() chains for FMA optimization
        uint k = tid * 4;
        for (; k + 3 < K; k += GROUP_SIZE * 4) {
            // Shared: load 4 F32 activations once
            uint4 x4 = src1.Load4(src1_base + k * 4);
            float x0 = asfloat(x4.x); float x1 = asfloat(x4.y);
            float x2 = asfloat(x4.z); float x3 = asfloat(x4.w);

            // Row 0 weights
#if NATIVE_FP16
            vector<float16_t,4> wh0 = src0.Load<vector<float16_t,4> >(src0_row0 + k * 2);
            acc0 = mad((float)wh0.x, x0, mad((float)wh0.y, x1,
                   mad((float)wh0.z, x2, mad((float)wh0.w, x3, acc0))));
#else
            uint2 w4_0 = src0.Load2(src0_row0 + k * 2);
            acc0 = mad(f16tof32(w4_0.x & 0xFFFFu), x0, mad(f16tof32(w4_0.x >> 16), x1,
                   mad(f16tof32(w4_0.y & 0xFFFFu), x2, mad(f16tof32(w4_0.y >> 16), x3, acc0))));
#endif

            // Row 1 weights
#if NATIVE_FP16
            vector<float16_t,4> wh1 = src0.Load<vector<float16_t,4> >(src0_row1 + k * 2);
            acc1 = mad((float)wh1.x, x0, mad((float)wh1.y, x1,
                   mad((float)wh1.z, x2, mad((float)wh1.w, x3, acc1))));
#else
            uint2 w4_1 = src0.Load2(src0_row1 + k * 2);
            acc1 = mad(f16tof32(w4_1.x & 0xFFFFu), x0, mad(f16tof32(w4_1.x >> 16), x1,
                   mad(f16tof32(w4_1.y & 0xFFFFu), x2, mad(f16tof32(w4_1.y >> 16), x3, acc1))));
#endif
        }
        // Remainder
        for (; k < K; k++) {
            float x = asfloat(src1.Load(src1_base + k * 4));
            acc0 = mad(load_auto(src0, src0_row0 + k * 2, 2), x, acc0);
            acc1 = mad(load_auto(src0, src0_row1 + k * 2, 2), x, acc1);
        }
    } else {
        // F32 weights: Load4 = 4 floats per iteration, mad() chains
        uint k = tid * 4;
        for (; k + 3 < K; k += GROUP_SIZE * 4) {
            uint4 x4 = src1.Load4(src1_base + k * 4);
            float x0 = asfloat(x4.x); float x1 = asfloat(x4.y);
            float x2 = asfloat(x4.z); float x3 = asfloat(x4.w);

            uint4 w0 = src0.Load4(src0_row0 + k * 4);
            acc0 = mad(asfloat(w0.x), x0, mad(asfloat(w0.y), x1,
                   mad(asfloat(w0.z), x2, mad(asfloat(w0.w), x3, acc0))));

            uint4 w1 = src0.Load4(src0_row1 + k * 4);
            acc1 = mad(asfloat(w1.x), x0, mad(asfloat(w1.y), x1,
                   mad(asfloat(w1.z), x2, mad(asfloat(w1.w), x3, acc1))));
        }
        for (; k < K; k++) {
            float x = asfloat(src1.Load(src1_base + k * 4));
            acc0 = mad(asfloat(src0.Load(src0_row0 + k * 4)), x, acc0);
            acc1 = mad(asfloat(src0.Load(src0_row1 + k * 4)), x, acc1);
        }
    }

    // Two-level reduction with tree reduction for cross-vendor correctness
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[32 + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = num_waves / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_acc[tid] += shared_acc[tid + s];
            shared_acc[32 + tid] += shared_acc[32 + tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        float result0 = shared_acc[0];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[32];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
