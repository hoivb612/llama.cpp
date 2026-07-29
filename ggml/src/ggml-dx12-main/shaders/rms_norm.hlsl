// rms_norm.hlsl - RMS Normalization
// dst[i] = src0[i] / sqrt(mean(src0^2) + eps)
// eps is stored in op_param_uint(0) as float
// Each thread group processes one row (ne0 elements)
// Uses wave intrinsics (SM 6.0) for efficient reduction
#include "ggml_common.hlsli"

// Wave-level reduction: one entry per wave. At the smallest supported wave
// width (8) a 256-thread group has 32 waves, so size for 32.
groupshared float wave_sums[32];

WAVE_SIZE_ATTR
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    // Decompose row into (i1, i2, i3)
    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    float eps = op_param_f32(0);
    uint local_id = gtid.x;
    // Use the RUNTIME wave size, not compile-time WARP_SIZE. Some drivers
    // (Intel iGPUs) ignore the forced [WaveSize(N)] and dispatch a wider
    // SIMD, which would alias multiple hardware waves onto one wave_sums
    // slot and race. WaveGetLaneCount() gives the true wave width.
    uint lane_count = WaveGetLaneCount();
    uint wave_count = (256 + lane_count - 1) / lane_count;
    uint wave_id = local_id / lane_count;

    // Compute sum of squares for this row
    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off, src0_esize);
        local_sum += val * val;
    }

    // Wave-level reduction
    float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wave_sums[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Cross-wave reduction: fold all wave partials in a single thread. A
    // second WaveActiveSum would only cover one hardware wave, so it breaks
    // when wave_count > lane_count (e.g. 32 waves of 8 lanes). Broadcast via
    // shared memory.
    if (local_id == 0) {
        float acc = wave_sums[0];
        for (uint w = 1; w < wave_count; w++) acc += wave_sums[w];
        wave_sums[0] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    float total = wave_sums[0];

    float rms = sqrt(total / (float)ne00 + eps);
    float scale_val = 1.0f / rms;

    // Normalize
    for (uint i0 = local_id; i0 < ne0; i0 += 256) {
        uint off_src = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_dst = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        float val = load_auto(src0, off_src, src0_esize);
        store_auto(dst, off_dst, val * scale_val, dst_esize);
    }
}
