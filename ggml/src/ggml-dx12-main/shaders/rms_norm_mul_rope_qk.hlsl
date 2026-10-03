// rms_norm_mul_rope_qk.hlsl - merged Q-norm and K-norm dispatch
//
// Runs the Qwen3-style QK-Norm chain for both the Q and K projections in a
// single dispatch. Groups [0, q_rows) handle Q (RMS_NORM + MUL + ROPE), groups
// [q_rows, total_rows) handle K (RMS_NORM + MUL + ROPE + VIEW + SET_ROWS).
// Restricted to ne02 == ne03 == 1 (decode), so a group's row index is also its
// head index within its region.
//
// Q region                         K region
//   src0 : activations (F32)         src6 : activations (F32, VA pre-offset)
//   src1 : q_norm weights            src5 : k_norm weights (VA pre-offset)
//   dst  : Qcur (F32)                temp : KV cache (VA pre-offset to layer)
//
// Shared: src2 ROPE positions (VA pre-offset), src3 SET_ROWS row indices
// (VA pre-offset), src4 ROPE freq_factors (VA pre-offset, when has_ff != 0).
//
// op_params[0]:  epsilon (float)
// op_params[1]:  n_dims (uint)
// op_params[2]:  mode (uint)
// op_params[3]:  corr_high (float)
// op_params[4]:  corr_low (float)
// op_params[5]:  freq_base (float)
// op_params[6]:  freq_scale (float)
// op_params[7]:  ext_factor (float)
// op_params[8]:  q_rows | (total_rows << 16)
// op_params[9]:  KV cache byte stride between rows
// op_params[10]: KV cache element size in bytes
// op_params[14]: attn_factor (float)
// op_params[15]: has_ff (uint)

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define BLOCK_SIZE 256

groupshared float wave_sums[32];
groupshared float norm_data[1024];

WAVE_SIZE_ATTR
[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row        = gid.x;
    uint q_rows     = op_param_uint(8) & 0xFFFFu;
    uint total_rows = op_param_uint(8) >> 16;
    if (row >= total_rows) return;

    bool is_k = row >= q_rows;
    uint i1   = is_k ? (row - q_rows) : row;

    uint local_id   = gtid.x;
    uint lane_count = WaveGetLaneCount();
    uint wave_count = (BLOCK_SIZE + lane_count - 1) / lane_count;
    uint wave_id    = local_id / lane_count;

    float eps = op_param_f32(0);

    // Phase 1: RMS_NORM
    precise float local_sum = 0.0f;
    for (uint i0 = local_id; i0 < ne00; i0 += BLOCK_SIZE) {
        uint off = i0 * nb00 + i1 * nb01;
        float val = asfloat(is_k ? src6.Load(off) : src0.Load(off + src0_offset));
        local_sum += val * val;
    }

    float ws = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) wave_sums[wave_id] = ws;
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        float acc = wave_sums[0];
        for (uint w = 1; w < wave_count; ++w) acc += wave_sums[w];
        wave_sums[0] = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    float scale_val = rsqrt(wave_sums[0] / (float)ne00 + eps);

    // Phase 2: Normalize + multiply by weight -> shared memory
    for (uint i0 = local_id; i0 < ne00; i0 += BLOCK_SIZE) {
        uint off_src = i0 * nb00 + i1 * nb01;
        uint off_wt  = (i0 % ne10) * nb10;
        float val = asfloat(is_k ? src6.Load(off_src) : src0.Load(off_src + src0_offset));
        float wt  = is_k ? load_auto(src5, off_wt, src1_esize)
                         : load_auto(src1, off_wt + src1_offset, src1_esize);
        if (i0 < 1024) norm_data[i0] = val * scale_val * wt;
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase 3: ROPE, then store to Qcur or scatter into the KV cache
    uint  n_dims      = op_param_uint(1);
    uint  mode        = op_param_uint(2);
    float freq_base   = op_param_f32(5);
    float freq_scale  = op_param_f32(6);
    float ext_factor  = op_param_f32(7);
    float corr_high   = op_param_f32(3);
    float corr_low    = op_param_f32(4);
    uint  kv_nb1      = op_param_uint(9);
    uint  kv_esize    = op_param_uint(10);
    float attn_factor = op_param_f32(14);
    uint  has_ff      = op_param_uint(15);

    bool is_neox   = (mode & 2u) != 0;
    uint half_dims = n_dims / 2;

    int pos = asint(src2.Load(0));

    uint kv_row_base = 0;
    if (is_k) {
        kv_row_base = (uint)asint(src3.Load(0)) * kv_nb1;
    }

    for (uint pair = local_id; pair < ne00 / 2; pair += BLOCK_SIZE) {
        uint idx_a, idx_b;

        if (pair >= half_dims) {
            uint pass_idx = n_dims + 2 * (pair - half_dims);
            for (uint p = 0; p < 2; ++p) {
                uint pi = pass_idx + p;
                if (pi >= ne00) continue;
                if (is_k) {
                    store_auto(temp, kv_row_base + (i1 * ne00 + pi) * kv_esize,
                               norm_data[pi], kv_esize);
                } else {
                    dst.Store(dst_offset + pi * nb0 + i1 * nb1, asuint(norm_data[pi]));
                }
            }
            continue;
        }

        if (is_neox) { idx_a = pair; idx_b = pair + half_dims; }
        else { idx_a = pair * 2; idx_b = pair * 2 + 1; }

        float theta_extrap = (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));
        if (has_ff != 0u) {
            theta_extrap = theta_extrap / asfloat(src4.Load(pair * 4));
        }

        float cos_theta, sin_theta;
        rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, pair, ext_factor, attn_factor, cos_theta, sin_theta);

        float x0 = norm_data[idx_a];
        float x1 = norm_data[idx_b];

        float rot_a = x0 * cos_theta - x1 * sin_theta;
        float rot_b = x0 * sin_theta + x1 * cos_theta;

        if (is_k) {
            store_auto(temp, kv_row_base + (i1 * ne00 + idx_a) * kv_esize, rot_a, kv_esize);
            store_auto(temp, kv_row_base + (i1 * ne00 + idx_b) * kv_esize, rot_b, kv_esize);
        } else {
            dst.Store(dst_offset + idx_a * nb0 + i1 * nb1, asuint(rot_a));
            dst.Store(dst_offset + idx_b * nb0 + i1 * nb1, asuint(rot_b));
        }
    }
}
