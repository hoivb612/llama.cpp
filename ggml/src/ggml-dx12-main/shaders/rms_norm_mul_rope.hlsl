// rms_norm_mul_rope.hlsl - 3-way fused op
// RMS_NORM + MUL + ROPE in one dispatch
// Mirrors rms_norm_mul_rope_set_rows.hlsl's structure (which is known to
// produce coherent output on Qwen3 K-side QK-Norm). The only differences
// vs the 5-way version are:
//   - no SET_ROWS row index (src3 is freq_factors here, not row_idx)
//   - dst write uses src0 strides (offset_4d) since output is normal
//     contiguous tensor, not a KV cache view
//
// src0: input to normalize (F32)
// src1: RMS norm weights (F32)
// src2: ROPE position indices (I32)
// src3: ROPE freq_factors (F32, optional - bound when has_ff != 0)
// dst:  rotated output (F32)
//
// op_params[0]: epsilon (float)
// op_params[1]: n_dims (uint)
// op_params[2]: mode (uint)
// op_params[3]: corr_high (float, host-precomputed YaRN range max)
// op_params[4]: corr_low  (float, host-precomputed YaRN range min)
// op_params[5]: freq_base (float)
// op_params[6]: freq_scale (float)
// op_params[7]: ext_factor (float)
// op_params[10]: ROPE position indices src offset
// op_params[12]: ROPE position indices nb0
// op_params[14]: attn_factor (float)
// op_params[15]: has_ff (uint)

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define BLOCK_SIZE 256

groupshared float wave_sums[16];
groupshared float norm_data[1024];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint total_rows = ne01 * ne02 * ne03;
    if (row >= total_rows) return;

    uint i3 = row / (ne01 * ne02);
    uint rem = row % (ne01 * ne02);
    uint i2 = rem / ne01;
    uint i1 = rem % ne01;

    uint local_id = gtid.x;
    uint wave_count = BLOCK_SIZE / WARP_SIZE;
    uint wave_id = local_id / WARP_SIZE;

    float eps = op_param_f32(0);

    // Phase 1: RMS_NORM
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

    // Phase 2: Normalize + multiply by weight -> shared memory
    for (uint i0 = local_id; i0 < ne00; i0 += BLOCK_SIZE) {
        uint off_src = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_wt = offset_4d(i0 % ne10, i1 % ne11, i2 % ne12, i3 % ne13,
                                nb10, nb11, nb12, nb13, src1_offset);
        float val = asfloat(src0.Load(off_src));
        float wt = load_auto(src1, off_wt, src1_esize);
        if (i0 < 1024) norm_data[i0] = val * scale_val * wt;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: ROPE
    uint  n_dims        = op_param_uint(1);
    uint  mode          = op_param_uint(2);
    float freq_base     = op_param_f32(5);
    float freq_scale    = op_param_f32(6);
    float ext_factor    = op_param_f32(7);
    float corr_high     = op_param_f32(3);
    float corr_low      = op_param_f32(4);
    uint  pos_offset    = op_param_uint(10);
    uint  pos_nb0       = op_param_uint(12);
    float attn_factor   = op_param_f32(14);
    uint  has_ff        = op_param_uint(15);

    bool is_neox = (mode & 2u) != 0;
    uint half_dims = n_dims / 2;

    int pos = asint(src2.Load(pos_offset + i2 * pos_nb0));

    for (uint pair = local_id; pair < ne00 / 2; pair += BLOCK_SIZE) {
        uint idx_a, idx_b;

        if (pair >= half_dims) {
            // Passthrough for partial-rotation models
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

        if (is_neox) { idx_a = pair; idx_b = pair + half_dims; }
        else { idx_a = pair * 2; idx_b = pair * 2 + 1; }

        float theta_extrap = (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));
        if (has_ff != 0u) {
            float ff = asfloat(src3.Load(pair * 4));
            theta_extrap = theta_extrap / ff;
        }

        float cos_theta, sin_theta;
        rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, pair, ext_factor, attn_factor, cos_theta, sin_theta);

        float x0 = norm_data[idx_a];
        float x1 = norm_data[idx_b];

        float rot_a = x0 * cos_theta - x1 * sin_theta;
        float rot_b = x0 * sin_theta + x1 * cos_theta;

        uint od_a = offset_4d(idx_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        uint od_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);

        dst.Store(od_a, asuint(rot_a));
        dst.Store(od_b, asuint(rot_b));
    }
}
