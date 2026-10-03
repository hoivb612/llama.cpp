// soft_max_cached.hlsl - Softmax for small rows (ne00 <= MAX_K).
// Identical math to soft_max.hlsl but caches the per-element value
// (scale*src0 + slope*mask) in LDS so we read src0/mask from global
// memory exactly once (instead of three times in the baseline).
//
// Stage 1 (load+max):     read src0/mask -> LDS pre[i], reduce max
// Stage 2 (exp+sum):      LDS pre[i] -> exp(.-max) -> LDS pre[i], reduce sum
// Stage 3 (normalize):    LDS pre[i] * inv_sum -> dst
//
// LDS: 1024 floats = 4 KiB (plus 64 bytes wave_max/wave_sum).
// Dispatched when ne00 <= MAX_K.
#include "ggml_common.hlsli"

#define MAX_K 1024
#define WG 256

groupshared float pre[MAX_K];
groupshared float wave_maxs[32];
groupshared float wave_sums[32];

WAVE_SIZE_ATTR
[numthreads(WG, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    float scale_val = op_param_f32(0);
    float max_bias  = op_param_f32(1);
    float m0        = op_param_f32(2);
    float m1        = op_param_f32(3);
    uint  n_head_log2 = op_param_uint(4);
    uint  has_sinks   = op_param_uint(5);
    uint  src2_off    = op_param_uint(6);

    uint local_id = gtid.x;
    bool has_mask = (ne10 > 0);
    uint lane_count = WaveGetLaneCount();
    uint wave_count = (WG + lane_count - 1) / lane_count;
    uint wave_id = local_id / lane_count;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        uint h = i2;
        float base = h < n_head_log2 ? m0 : m1;
        uint  exp_val = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;
        slope = pow(base, (float)exp_val);
    }

    uint mask_i2 = ne12 > 0 ? (i2 % ne12) : 0;
    uint mask_i3 = ne13 > 0 ? (i3 % ne13) : 0;

    // Stage 1: load values into LDS, compute local max.
    float local_max = -3.402823466e+38f;
    for (uint i0 = local_id; i0 < ne00; i0 += WG) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off, src0_esize) * scale_val;
        if (has_mask) {
            uint mask_off = offset_4d(i0, i1, mask_i2, mask_i3, nb10, nb11, nb12, nb13, src1_offset);
            val += slope * load_auto(src1, mask_off, src1_esize);
        }
        pre[i0] = val;
        local_max = max(local_max, val);
    }

    float sink_val = 0.0f;
    if (has_sinks != 0) {
        sink_val = asfloat(src2.Load(src2_off + i2 * 4));
        local_max = max(local_max, sink_val);
    }

    float wave_max = WaveActiveMax(local_max);
    if (WaveIsFirstLane()) { wave_maxs[wave_id] = wave_max; }
    GroupMemoryBarrierWithGroupSync();

    float row_max = -3.402823466e+38f;
    if (local_id == 0) {
        float acc = wave_maxs[0];
        for (uint w = 1; w < wave_count; ++w) acc = max(acc, wave_maxs[w]);
        wave_maxs[0] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    row_max = wave_maxs[0];

    // Stage 2: compute exp(val - row_max) into LDS, accumulate sum.
    precise float local_sum = 0.0f;
    for (uint i0b = local_id; i0b < ne00; i0b += WG) {
        float e = exp(pre[i0b] - row_max);
        pre[i0b] = e;
        local_sum += e;
    }

    float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) { wave_sums[wave_id] = wave_sum; }
    GroupMemoryBarrierWithGroupSync();

    float total_sum = 0.0f;
    if (local_id == 0) {
        float acc = wave_sums[0];
        for (uint w = 1; w < wave_count; ++w) acc += wave_sums[w];
        wave_sums[0] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    total_sum = wave_sums[0];

    if (has_sinks != 0) {
        total_sum += exp(sink_val - row_max);
    }
    float inv_sum = 1.0f / total_sum;

    // Stage 3: normalize and store. No global reads of src0/mask.
    for (uint i0c = local_id; i0c < ne0; i0c += WG) {
        uint off_dst = offset_4d(i0c, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_dst, pre[i0c] * inv_sum, dst_esize);
    }
}
