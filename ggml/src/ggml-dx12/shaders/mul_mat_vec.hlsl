// mul_mat_vec.hlsl - Specialized matrix-vector multiply (M=1) using K-reduction
// ggml MUL_MAT: dst[row, 0, i2, i3] = sum_k(src0[k, row, i2_src0, i3_src0] * src1[k, 0, i2, i3])
//
// src0: weights, ne00 = K, ne01 = N — F16 or F32
// src1: input,   ne10 = K, ne11 = 1 — F32
// dst:  output,  ne0  = N, ne1  = 1 — F32
//
// Dispatch: groups_x = N (one group per output row), groups_y = 1, groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE 256

groupshared float partial[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint row = gid.x;

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;

    uint src0_base = src0_offset + row * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    if (src0_esize == 2) {
        // F16 weights: vectorized 2-element loads
        // Input is always contiguous F32 during generation (nb10=4)
        uint k = tid * 2;
        if (nb10 == 4) {
            // Fast path: paired Load2 for contiguous F32 input
            for (; k + 1 < K; k += GROUP_SIZE * 2) {
                uint w2 = src0.Load(src0_base + k * 2);
                float w0 = f16tof32(w2 & 0xFFFFu);
                float w1 = f16tof32(w2 >> 16);
                uint2 xp = src1.Load2(src1_base + k * 4);
                acc += w0 * asfloat(xp.x) + w1 * asfloat(xp.y);
            }
        } else {
            for (; k + 1 < K; k += GROUP_SIZE * 2) {
                uint w2 = src0.Load(src0_base + k * 2);
                float w0 = f16tof32(w2 & 0xFFFFu);
                float w1 = f16tof32(w2 >> 16);
                float x0 = load_auto(src1, src1_base + k * nb10, src1_esize);
                float x1 = load_auto(src1, src1_base + (k + 1) * nb10, src1_esize);
                acc += w0 * x0 + w1 * x1;
            }
        }
        if (k < K) {
            float w = load_auto(src0, src0_base + k * 2, src0_esize);
            float x = load_auto(src1, src1_base + k * nb10, src1_esize);
            acc += w * x;
        }
    } else {
        // F32 weights: use Load2 for paired reads when possible
        if (nb10 == 4) {
            uint k = tid * 2;
            for (; k + 1 < K; k += GROUP_SIZE * 2) {
                uint2 wp = src0.Load2(src0_base + k * 4);
                uint2 xp = src1.Load2(src1_base + k * 4);
                acc += asfloat(wp.x) * asfloat(xp.x) + asfloat(wp.y) * asfloat(xp.y);
            }
            if (k < K) {
                acc += asfloat(src0.Load(src0_base + k * 4)) * asfloat(src1.Load(src1_base + k * 4));
            }
        } else {
            for (uint k = tid; k < K; k += GROUP_SIZE) {
                float w = asfloat(src0.Load(src0_base + k * 4));
                float x = load_auto(src1, src1_base + k * nb10, src1_esize);
                acc += w * x;
            }
        }
    }

    // Hybrid reduction: wave intrinsics first, then cross-wave via shared memory
    // Intel Arc wave size = 16, so 256 threads = 16 waves
    float wave_sum = WaveActiveSum(acc);

    uint wave_id = tid / WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint num_waves = GROUP_SIZE / WaveGetLaneCount();

    // Each wave's first lane writes its sum
    if (lane_id == 0) {
        partial[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // First wave reduces across all waves (1 barrier instead of 8)
    if (tid < num_waves) {
        float cross_wave = partial[tid];
        cross_wave = WaveActiveSum(cross_wave);
        if (tid == 0) {
            partial[0] = cross_wave;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0 && row < ne0) {
        float result = partial[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + row * 4));
        uint off_d = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
