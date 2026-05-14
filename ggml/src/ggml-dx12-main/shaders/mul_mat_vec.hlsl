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
    uint row = group_x_2d(gid);
    if (row >= ne0) return;

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;

    uint src0_base = src0_offset + row * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;
    bool src0_pair_aligned = (src0_base & 3u) == 0;

    precise float acc = 0.0f;

    if (src0_esize == 2 && nb10 == 4 && src0_pair_aligned) {
        // F16 weights + contiguous F32 input: paired loads + mad()
        uint k = tid * 2;
        for (; k + 1 < K; k += GROUP_SIZE * 2) {
#if NATIVE_FP16
            // Native 16-bit path: templated Load emits native HW fp16 unpack
            // (no f16tof32). Multiply against F32 input — accumulator stays F32
            // to preserve precision (no f16acc opt-in here).
            vector<float16_t,2> wh = src0.Load<vector<float16_t,2> >(src0_base + k * 2);
            uint2 xp = src1.Load2(src1_base + k * 4);
            acc = mad((float)wh.x, asfloat(xp.x), mad((float)wh.y, asfloat(xp.y), acc));
#else
            uint w2 = src0.Load(src0_base + k * 2);
            float w0 = f16tof32(w2 & 0xFFFFu);
            float w1 = f16tof32(w2 >> 16);
            uint2 xp = src1.Load2(src1_base + k * 4);
            acc = mad(w0, asfloat(xp.x), mad(w1, asfloat(xp.y), acc));
#endif
        }
        if (k < K) {
            acc += load_auto(src0, src0_base + k * 2, 2) * asfloat(src1.Load(src1_base + k * 4));
        }
    } else if (src0_esize == 3 && nb10 == 4 && src0_pair_aligned) {
        // BF16 weights (physical stride 2) + contiguous F32 input: paired loads + mad()
        uint k = tid * 2;
        for (; k + 1 < K; k += GROUP_SIZE * 2) {
            uint w2 = src0.Load(src0_base + k * 2);
            float w0 = asfloat((w2 & 0xFFFFu) << 16);
            float w1 = asfloat((w2 & 0xFFFF0000u));
            uint2 xp = src1.Load2(src1_base + k * 4);
            acc = mad(w0, asfloat(xp.x), mad(w1, asfloat(xp.y), acc));
        }
        if (k < K) {
            acc += load_auto(src0, src0_base + k * 2, 3) * asfloat(src1.Load(src1_base + k * 4));
        }
    } else if (src0_esize == 3) {
        // BF16 weights, non-contiguous input
        if (src0_pair_aligned) {
            uint k = tid * 2;
            for (; k + 1 < K; k += GROUP_SIZE * 2) {
                uint w2 = src0.Load(src0_base + k * 2);
                float w0 = asfloat((w2 & 0xFFFFu) << 16);
                float w1 = asfloat((w2 & 0xFFFF0000u));
                float x0 = load_auto(src1, src1_base + k * nb10, src1_esize);
                float x1 = load_auto(src1, src1_base + (k + 1) * nb10, src1_esize);
                acc = mad(w0, x0, mad(w1, x1, acc));
            }
            if (k < K) {
                acc += load_auto(src0, src0_base + k * 2, 3) *
                       load_auto(src1, src1_base + k * nb10, src1_esize);
            }
        } else {
            for (uint k = tid; k < K; k += GROUP_SIZE) {
                acc += load_auto(src0, src0_base + k * 2, 3) *
                       load_auto(src1, src1_base + k * nb10, src1_esize);
            }
        }
    } else if (src0_esize == 2) {
        // F16 weights, non-contiguous input
        if (src0_pair_aligned) {
            uint k = tid * 2;
            for (; k + 1 < K; k += GROUP_SIZE * 2) {
                uint w2 = src0.Load(src0_base + k * 2);
                float w0 = f16tof32(w2 & 0xFFFFu);
                float w1 = f16tof32(w2 >> 16);
                float x0 = load_auto(src1, src1_base + k * nb10, src1_esize);
                float x1 = load_auto(src1, src1_base + (k + 1) * nb10, src1_esize);
                acc = mad(w0, x0, mad(w1, x1, acc));
            }
            if (k < K) {
                acc += load_auto(src0, src0_base + k * 2, 2) *
                       load_auto(src1, src1_base + k * nb10, src1_esize);
            }
        } else {
            for (uint k = tid; k < K; k += GROUP_SIZE) {
                acc += load_auto(src0, src0_base + k * 2, 2) *
                       load_auto(src1, src1_base + k * nb10, src1_esize);
            }
        }
    } else if (nb10 == 4) {
        // F32 weights + contiguous F32 input: paired loads + mad()
        uint k = tid * 2;
        for (; k + 1 < K; k += GROUP_SIZE * 2) {
            uint2 wp = src0.Load2(src0_base + k * 4);
            uint2 xp = src1.Load2(src1_base + k * 4);
            acc = mad(asfloat(wp.x), asfloat(xp.x), mad(asfloat(wp.y), asfloat(xp.y), acc));
        }
        if (k < K) {
            acc += asfloat(src0.Load(src0_base + k * 4)) * asfloat(src1.Load(src1_base + k * 4));
        }
    } else {
        for (uint k = tid; k < K; k += GROUP_SIZE) {
            float w = asfloat(src0.Load(src0_base + k * 4));
            float x = load_auto(src1, src1_base + k * nb10, src1_esize);
            acc = mad(w, x, acc);
        }
    }

    // Hybrid reduction: wave intrinsics first, then cross-wave via shared memory
    float wave_sum = WaveActiveSum(acc);

    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;

    if (WaveIsFirstLane()) {
        partial[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (num_waves <= WARP_SIZE) {
        if (tid < num_waves) {
            float v = partial[tid];
            v = WaveActiveSum(v);
            if (tid == 0) partial[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
    } else {
        for (uint s = num_waves / 2; s > 0; s /= 2) {
            if (tid < s) partial[tid] += partial[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (tid == 0 && row < ne0) {
        float result = partial[0];
        result += load_fused_bias(row, i2, i3);
        uint off_d = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}
