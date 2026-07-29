// mul_mat_vec_mr32.hlsl - Compact multi-row F16/F32 matvec (M=1)
//
// 32 threads per workgroup (matching Vulkan's subgroup-sized approach).
// Each thread processes many K elements, giving better ILP and register
// utilization than the 256-thread variant on small-K models.
// Processes 2 output rows per workgroup, sharing activation loads.
//
// Dispatch: groups_x = (N+1)/2, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 32
#define NUM_ROWS   2

groupshared float shared_acc[8];  // max 4 waves (min lane count 8) x 2 rows

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
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

    bool k_contig = (nb00 == src0_esize) && (nb10 == 4u);

    if (src0_esize == 2 && k_contig) {
        // F16 weights: 4 elements per iteration using Load2 for weights + Load4 for activations
        uint k = tid * 4;
        for (; k + 3 < K; k += GROUP_SIZE * 4) {
            uint4 x4 = src1.Load4(src1_base + k * 4);
            float x0 = asfloat(x4.x); float x1 = asfloat(x4.y);
            float x2 = asfloat(x4.z); float x3 = asfloat(x4.w);

#if NATIVE_FP16
            vector<float16_t,4> wh0 = src0.Load<vector<float16_t,4> >(src0_row0 + k * 2);
            acc0 = mad((float)wh0.x, x0, mad((float)wh0.y, x1,
                   mad((float)wh0.z, x2, mad((float)wh0.w, x3, acc0))));
            vector<float16_t,4> wh1 = src0.Load<vector<float16_t,4> >(src0_row1 + k * 2);
            acc1 = mad((float)wh1.x, x0, mad((float)wh1.y, x1,
                   mad((float)wh1.z, x2, mad((float)wh1.w, x3, acc1))));
#else
            uint2 w0 = src0.Load2(src0_row0 + k * 2);
            acc0 = mad(f16tof32(w0.x & 0xFFFFu), x0, mad(f16tof32(w0.x >> 16), x1,
                   mad(f16tof32(w0.y & 0xFFFFu), x2, mad(f16tof32(w0.y >> 16), x3, acc0))));

            uint2 w1 = src0.Load2(src0_row1 + k * 2);
            acc1 = mad(f16tof32(w1.x & 0xFFFFu), x0, mad(f16tof32(w1.x >> 16), x1,
                   mad(f16tof32(w1.y & 0xFFFFu), x2, mad(f16tof32(w1.y >> 16), x3, acc1))));
#endif
        }
        for (; k < K; k++) {
            float x = asfloat(src1.Load(src1_base + k * 4));
            acc0 = mad(load_auto(src0, src0_row0 + k * 2, 2), x, acc0);
            acc1 = mad(load_auto(src0, src0_row1 + k * 2, 2), x, acc1);
        }
    } else if (src0_esize == 4 && k_contig) {
        // F32 weights
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
    } else {
        // Strided fallback: per-element scalar loads using nb00 / nb10.
        for (uint k = tid; k < K; k += GROUP_SIZE) {
            float x = asfloat(src1.Load(src1_base + k * nb10));
            acc0 = mad(load_auto(src0, src0_row0 + k * nb00, src0_esize), x, acc0);
            acc1 = mad(load_auto(src0, src0_row1 + k * nb00, src0_esize), x, acc1);
        }
    }

    // Wave reduction. Use the RUNTIME wave size (WaveGetLaneCount) for slot
    // indexing rather than compile-time WARP_SIZE: some drivers (Intel iGPUs)
    // ignore the forced [WaveSize(N)] and run a wider SIMD, which would alias
    // two hardware waves onto one shared_acc slot and race. row1 uses a fixed
    // base of 4 (= GROUP_SIZE / min-lane-count 8) so the two rows never
    // overlap regardless of the actual lane count.
    float wave_sum0 = WaveActiveSum(acc0);
    float wave_sum1 = WaveActiveSum(acc1);
    uint lane_count = WaveGetLaneCount();
    uint wave_id = tid / lane_count;
    uint num_waves = (GROUP_SIZE + lane_count - 1) / lane_count;
    const uint row1_off = 4;  // GROUP_SIZE / min-lane-count(8)

    if (WaveIsFirstLane()) {
        shared_acc[wave_id] = wave_sum0;
        shared_acc[row1_off + wave_id] = wave_sum1;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result0 = shared_acc[0];
        for (uint w = 1; w < num_waves; w++) result0 += shared_acc[w];
        result0 += load_fused_bias(row0, i2, i3);
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);

        if (row0 + 1 < ne0) {
            float result1 = shared_acc[row1_off];
            for (uint w = 1; w < num_waves; w++) result1 += shared_acc[row1_off + w];
            result1 += load_fused_bias(row0 + 1, i2, i3);
            uint off_d1 = offset_4d(row0 + 1, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off_d1, result1, dst_esize);
        }
    }
}
