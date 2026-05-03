// rms_norm_mul_rope.hlsl - Fused RMS_NORM + MUL + ROPE
// Normalizes, multiplies by weight, then applies rotary embedding in one dispatch
// Eliminates 1 dispatch (ROPE) per layer
//
// src0: input to normalize (F32)
// src1: RMS norm weights (F32)
// src2: ROPE position indices (I32)
// dst:  rotated output (F32)
//
// op_params[0]: epsilon (float)
// op_params[1..7]: ROPE params (n_dims, mode, freq_base, freq_scale, etc.)
//   [1]=n_dims, [2]=mode, [5]=freq_base(float), [6]=freq_scale(float)

#include "ggml_common.hlsli"

#define BLOCK_SIZE 256

groupshared float wave_sums[16];
groupshared float norm_data[1024];  // max ne00 for shared memory pass to ROPE

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne1 * ne2 * ne3;
    if (row >= total_rows) return;

    uint i3 = row / (ne1 * ne2);
    uint rem = row % (ne1 * ne2);
    uint i2 = rem / ne1;
    uint i1 = rem % ne1;

    uint local_id = gtid.x;
    uint wave_count = BLOCK_SIZE / WaveGetLaneCount();
    uint wave_id = local_id / WaveGetLaneCount();

    float eps = op_param_f32(0);

    // Phase 1: RMS_NORM — compute sum of squares
    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += BLOCK_SIZE) {
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        float val = asfloat(src0.Load(off));
        local_sum += val * val;
    }

    float ws = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) wave_sums[wave_id] = ws;
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    if (local_id < wave_count) total = wave_sums[local_id];
    total = WaveActiveSum(total);
    if (local_id == 0) wave_sums[0] = total;
    GroupMemoryBarrierWithGroupSync();
    total = wave_sums[0];

    float scale_val = rsqrt(total / (float)ne00 + eps);

    // Phase 2: Normalize, multiply by weight, store to shared memory for ROPE
    for (uint i0 = local_id; i0 < ne00; i0 += BLOCK_SIZE) {
        uint off_src = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_wt = src1_offset + i0 * 4;  // weights are contiguous F32
        float val = asfloat(src0.Load(off_src));
        float wt = asfloat(src1.Load(off_wt));
        float normed = val * scale_val * wt;
        if (i0 < 1024) norm_data[i0] = normed;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: ROPE — apply rotary position embedding
    uint n_dims = op_param_uint(1);
    uint mode = op_param_uint(2);
    float freq_base = op_param_f32(5);
    float freq_scale = op_param_f32(6);

    bool is_neox = (mode & 2u) != 0;
    uint half_dims = n_dims / 2;

    // Position from src2 (int32)
    uint pos_off = i2 * 4;  // src2 offset for this position
    int pos = asint(src2.Load(pos_off));

    for (uint pair = local_id; pair < ne00 / 2; pair += BLOCK_SIZE) {
        uint idx_a, idx_b;

        if (pair >= half_dims) {
            // Passthrough: copy from shared mem to output
            uint pass_idx = n_dims + 2 * (pair - half_dims);
            if (pass_idx < ne00) {
                uint od = offset_4d(pass_idx, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                dst.Store(od, asuint(norm_data[pass_idx]));
            }
            if (pass_idx + 1 < ne00) {
                uint od = offset_4d(pass_idx + 1, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                dst.Store(od, asuint(norm_data[pass_idx + 1]));
            }
            continue;
        }

        if (is_neox) {
            idx_a = pair;
            idx_b = pair + half_dims;
        } else {
            idx_a = pair * 2;
            idx_b = pair * 2 + 1;
        }

        float theta = (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));
        theta *= freq_scale;

        float cos_theta, sin_theta;
        sincos(theta, sin_theta, cos_theta);

        float x0 = norm_data[idx_a];
        float x1 = norm_data[idx_b];

        uint od_a = offset_4d(idx_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        uint od_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        dst.Store(od_a, asuint(x0 * cos_theta - x1 * sin_theta));
        dst.Store(od_b, asuint(x0 * sin_theta + x1 * cos_theta));
    }
}
