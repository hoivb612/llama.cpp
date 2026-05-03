// add_rms_norm_mul.hlsl - Fused ADD + RMS_NORM + MUL (triple fusion)
// Pattern: residual = src0 + src1; norm = rms_norm(residual); out = norm * weight
// Eliminates 2 dispatches per transformer layer
//
// src0: first ADD input (e.g., previous residual)
// src1: second ADD input (e.g., attention/FFN output)
// src2: RMS norm weight tensor
// dst:  UAV for both ADD intermediate and final output
//
// op_params[0]: ADD dst offset (to store intermediate sum)
// op_params[1]: weight (src2) offset
// op_params[2]: epsilon (as float bits)
// op_params[3]: ADD dst esize (2=F16, 4=F32)

#include "ggml_common.hlsli"

#define MAX_CACHED 16  // max elements per thread (ne00/256, covers up to 4096)

groupshared float wave_sums[16];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    uint local_id = gtid.x;
    uint wave_count = 256 / WaveGetLaneCount();
    uint wave_id = local_id / WaveGetLaneCount();

    uint add_dst_off = op0;
    uint wt_offset = op1;
    float eps = asfloat(op2);
    uint add_dst_esize = op3;

    // Pass 1: Compute sum, store ADD intermediate, cache in registers
    float cached[MAX_CACHED];
    uint n_cached = 0;
    precise float local_sq_sum = 0.0f;

    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off_a = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_b = offset_4d(i0, i1, i2, i3, nb10, nb11, nb12, nb13, src1_offset);
        float a = load_auto(src0, off_a, src0_esize);
        float b = load_auto(src1, off_b, src1_esize);
        float sum = a + b;

        // Store ADD intermediate for later layers
        uint off_add = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, add_dst_off);
        store_auto(dst, off_add, sum, add_dst_esize);

        // Cache in registers to avoid UAV read-back
        if (n_cached < MAX_CACHED) cached[n_cached++] = sum;

        local_sq_sum += sum * sum;
    }

    // Reduce sum-of-squares
    float wave_sum = WaveActiveSum(local_sq_sum);
    if (WaveIsFirstLane()) wave_sums[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (local_id < wave_count) total = wave_sums[local_id];
    total = WaveActiveSum(total);
    if (local_id == 0) wave_sums[0] = total;
    GroupMemoryBarrierWithGroupSync();
    total = wave_sums[0];

    float scale_val = rsqrt(total / (float)ne00 + eps);

    // Pass 2: Normalize and multiply by weight, using cached sums
    uint ci = 0;
    for (uint i0 = local_id; i0 < ne0; i0 += 256) {
        float sum = (ci < n_cached) ? cached[ci++] : 0.0f;
        float wt = asfloat(src2.Load(wt_offset + i0 * 4));
        uint off_dst = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_dst, sum * scale_val * wt, dst_esize);
    }
}
