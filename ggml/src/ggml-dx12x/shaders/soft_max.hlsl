// soft_max.hlsl - Softmax
// dst = softmax(src0 * scale + slope * mask)
// op_params: [0]=scale, [1]=max_bias, [2]=m0, [3]=m1, [4]=n_head_log2,
//            [5]=has_sinks, [6]=src2_offset
// src0 = input, src1 = mask (optional), src2 = sinks (optional)
// Each thread group processes one row
// Uses wave intrinsics (SM 6.0) for efficient reduction
#include "ggml_common.hlsli"

groupshared float wave_maxs[16];
groupshared float wave_sums[16];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
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
    uint wave_count = 256 / WaveGetLaneCount();
    uint wave_id = local_id / WaveGetLaneCount();

    // ALiBi slope
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        uint h = i2; // head index
        float base = h < n_head_log2 ? m0 : m1;
        uint  exp_val = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;
        slope = pow(base, (float)exp_val);
    }

    // Mask broadcasting: match CPU/Vulkan i02%ne12, i03%ne13
    uint mask_i2 = ne12 > 0 ? (i2 % ne12) : 0;
    uint mask_i3 = ne13 > 0 ? (i3 % ne13) : 0;

    // Find max value in row
    float local_max = -3.402823466e+38f; // -FLT_MAX
    for (uint i0 = local_id; i0 < ne00; i0 += 256) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off, src0_esize) * scale_val;
        if (has_mask) {
            uint mask_off = offset_4d(i0, i1, mask_i2, mask_i3, nb10, nb11, nb12, nb13, src1_offset);
            val += slope * load_auto(src1, mask_off, src1_esize);
        }
        local_max = max(local_max, val);
    }

    // If sinks, include the sink value in the max
    float sink_val = 0.0f;
    if (has_sinks != 0) {
        sink_val = asfloat(src2.Load(src2_off + i2 * 4));
        local_max = max(local_max, sink_val);
    }

    // Wave-level max reduction
    float wave_max = WaveActiveMax(local_max);
    if (WaveIsFirstLane()) {
        wave_maxs[wave_id] = wave_max;
    }
    GroupMemoryBarrierWithGroupSync();

    // Cross-wave max reduction + broadcast
    float row_max = -3.402823466e+38f;
    if (local_id < wave_count) {
        row_max = wave_maxs[local_id];
    }
    row_max = WaveActiveMax(row_max);
    if (local_id == 0) { wave_maxs[0] = row_max; }
    GroupMemoryBarrierWithGroupSync();
    row_max = wave_maxs[0];

    // Compute exp and sum
    precise float local_sum = 0.0f;
    for (uint i0b = local_id; i0b < ne00; i0b += 256) {
        uint off = offset_4d(i0b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = load_auto(src0, off, src0_esize) * scale_val;
        if (has_mask) {
            uint mask_off = offset_4d(i0b, i1, mask_i2, mask_i3, nb10, nb11, nb12, nb13, src1_offset);
            val += slope * load_auto(src1, mask_off, src1_esize);
        }
        local_sum += exp(val - row_max);
    }

    // Wave-level sum reduction
    float wave_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wave_sums[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Cross-wave sum reduction + broadcast
    float total_sum = 0.0f;
    if (local_id < wave_count) {
        total_sum = wave_sums[local_id];
    }
    total_sum = WaveActiveSum(total_sum);
    if (local_id == 0) { wave_sums[0] = total_sum; }
    GroupMemoryBarrierWithGroupSync();
    total_sum = wave_sums[0];

    // Include sink in sum
    if (has_sinks != 0) {
        total_sum += exp(sink_val - row_max);
    }
    float inv_sum = 1.0f / total_sum;

    // Write normalized output
    for (uint i0c = local_id; i0c < ne0; i0c += 256) {
        uint off_src = offset_4d(i0c, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_dst = offset_4d(i0c, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        float val = load_auto(src0, off_src, src0_esize) * scale_val;
        if (has_mask) {
            uint mask_off = offset_4d(i0c, i1, mask_i2, mask_i3, nb10, nb11, nb12, nb13, src1_offset);
            val += slope * load_auto(src1, mask_off, src1_esize);
        }
        store_auto(dst, off_dst, exp(val - row_max) * inv_sum, dst_esize);
    }
}
