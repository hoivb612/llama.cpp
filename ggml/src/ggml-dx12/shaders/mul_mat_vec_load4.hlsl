// mul_mat_vec_load4.hlsl - F16 matvec with Load4 vectorized reads
// Loads 4 F16 weights (2 uint32s = 4 F16) and 4 F32 inputs per iteration
// Better memory throughput than Load2 variant for large K

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
        // F16 weights: Load4 = 2 uint32 = 4 F16 values per iteration
        uint k = tid * 4;
        for (; k + 3 < K; k += GROUP_SIZE * 4) {
            uint2 w4 = src0.Load2(src0_base + k * 2);
            float w0 = f16tof32(w4.x & 0xFFFFu);
            float w1 = f16tof32(w4.x >> 16);
            float w2 = f16tof32(w4.y & 0xFFFFu);
            float w3 = f16tof32(w4.y >> 16);

            uint4 x4 = src1.Load4(src1_base + k * 4);
            acc += w0 * asfloat(x4.x) + w1 * asfloat(x4.y) + w2 * asfloat(x4.z) + w3 * asfloat(x4.w);
        }
        // Handle remainder
        for (; k < K; k++) {
            float w = load_auto(src0, src0_base + k * 2, 2);
            float x = asfloat(src1.Load(src1_base + k * 4));
            acc += w * x;
        }
    } else {
        // F32 weights: Load4 = 4 floats per iteration
        uint k = tid * 4;
        for (; k + 3 < K; k += GROUP_SIZE * 4) {
            uint4 w4 = src0.Load4(src0_base + k * 4);
            uint4 x4 = src1.Load4(src1_base + k * 4);
            acc += asfloat(w4.x) * asfloat(x4.x) + asfloat(w4.y) * asfloat(x4.y)
                 + asfloat(w4.z) * asfloat(x4.z) + asfloat(w4.w) * asfloat(x4.w);
        }
        for (; k < K; k++) {
            acc += asfloat(src0.Load(src0_base + k * 4)) * asfloat(src1.Load(src1_base + k * 4));
        }
    }

    // Wave-intrinsic reduction then cross-wave via shared memory
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint num_waves = GROUP_SIZE / WaveGetLaneCount();

    if (lane_id == 0) partial[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    if (tid < num_waves) {
        float cross_wave = partial[tid];
        cross_wave = WaveActiveSum(cross_wave);
        if (tid == 0) partial[0] = cross_wave;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0 && row < ne0) {
        float result = partial[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + row * 4));
        uint off_d = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
