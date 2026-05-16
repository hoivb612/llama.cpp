// norm.hlsl - Layer Normalization
// dst = (src0 - mean) / sqrt(var + eps)
// eps in op_param_uint(0)
// Uses wave intrinsics (SM 6.0) for efficient reduction
#include "ggml_common.hlsli"

groupshared float wave_vals[16];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    float eps = op_param_f32(0);
    uint local_id = gtid.x;
    uint wave_count = 256 / WaveGetLaneCount();
    uint wave_id = local_id / WaveGetLaneCount();

    // Compute mean - per-thread partial sum
    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        local_sum += load_auto(src0, off, src0_esize);
    }

    // Wave-level reduction for mean
    float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wave_vals[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (local_id < wave_count) {
        total = wave_vals[local_id];
    }
    total = WaveActiveSum(total);
    if (local_id == 0) { wave_vals[0] = total; }
    GroupMemoryBarrierWithGroupSync();
    total = wave_vals[0];
    float mean = total / (float)ne00;

    // Compute variance - per-thread partial sum
    precise float local_var = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float diff = load_auto(src0, off, src0_esize) - mean;
        local_var += diff * diff;
    }

    // Wave-level reduction for variance
    float wave_var = WaveActiveSum(local_var);
    if (WaveIsFirstLane()) {
        wave_vals[wave_id] = wave_var;
    }
    GroupMemoryBarrierWithGroupSync();

    float total_var = 0.0f;
    if (local_id < wave_count) {
        total_var = wave_vals[local_id];
    }
    total_var = WaveActiveSum(total_var);
    if (local_id == 0) { wave_vals[0] = total_var; }
    GroupMemoryBarrierWithGroupSync();
    total_var = wave_vals[0];
    float inv_std = 1.0f / sqrt(total_var / (float)ne00 + eps);

    // Normalize
    for (uint i0 = local_id; i0 < ne0; i0 += 256) {
        uint off_src = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_dst = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_dst, (load_auto(src0, off_src, src0_esize) - mean) * inv_std, dst_esize);
    }
}
