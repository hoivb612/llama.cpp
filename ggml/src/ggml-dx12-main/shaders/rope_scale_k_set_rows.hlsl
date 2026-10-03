// Fused Q RoPE + scale and K RoPE + KV-cache scatter.
//
// src0: Q input (F32)
// src1: shared position indices (I32)
// src2: shared frequency factors (F32, optional)
// src3: K input (F32, tensor offset baked into the binding)
// src4: K SET_ROWS indices (I32/I64, tensor offset baked into the binding)
// dst:  scaled Q output (F32)
// temp: K cache (F16/F32)
//
// op_params:
//   [1]=n_dims, [2]=mode, [3]=corr_high, [4]=corr_low
//   [5]=freq_base, [6]=freq_scale, [7]=ext_factor
//   [8]=Q scale
//   [9]=K cache base offset, [10]=K cache nb0, [11]=K cache nb1
//   [12]=K cache element size
//   [14]=attn_factor, [15]=has_ff

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint n_pairs = ne00 / 2;
    uint total_pairs = n_pairs * ne01 * ne02 * ne03;
    if (idx >= total_pairs) return;

    uint i3 = idx / (n_pairs * ne01 * ne02);
    uint rem = idx % (n_pairs * ne01 * ne02);
    uint i2 = rem / (n_pairs * ne01);
    rem %= n_pairs * ne01;
    uint i1 = rem / n_pairs;
    uint pair = rem % n_pairs;

    uint  n_dims      = op1;
    uint  mode        = op2;
    float corr_high   = asfloat(op3);
    float corr_low    = asfloat(op4);
    float freq_base   = asfloat(op5);
    float freq_scale  = asfloat(op6);
    float ext_factor  = asfloat(op7);
    float q_scale     = asfloat(op8);
    float attn_factor = asfloat(op14);
    uint  has_ff      = op15;

    bool is_neox = (mode & 2u) != 0;
    uint half_dims = n_dims / 2;
    uint idx_a;
    uint idx_b;
    if (pair >= half_dims) {
        idx_a = n_dims + 2 * (pair - half_dims);
        idx_b = idx_a + 1;
    } else if (is_neox) {
        idx_a = pair;
        idx_b = pair + half_dims;
    } else {
        idx_a = pair * 2;
        idx_b = idx_a + 1;
    }

    int row_idx = asint(src4.Load(0));

    float cos_theta = 1.0f;
    float sin_theta = 0.0f;
    if (pair < half_dims) {
        int pos = asint(src1.Load(src1_offset + i2 * nb10));
        float theta_extrap =
            (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));
        if (has_ff != 0u) {
            theta_extrap /= asfloat(src2.Load(pair * 4));
        }
        rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, pair,
                  ext_factor, attn_factor, cos_theta, sin_theta);
    }

    if (idx_a < ne00) {
        uint q_off_a = offset_4d(idx_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint k_off_a = offset_4d(idx_a, i1, i2, i3, nb00, nb01, nb02, nb03, 0);
        float q0 = asfloat(src0.Load(q_off_a));
        float k0 = asfloat(src3.Load(k_off_a));
        float q_out_a = q0;
        float k_out_a = k0;

        if (pair < half_dims && idx_b < ne00) {
            uint q_off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint k_off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, 0);
            float q1 = asfloat(src0.Load(q_off_b));
            float k1 = asfloat(src3.Load(k_off_b));
            q_out_a = q0 * cos_theta - q1 * sin_theta;
            k_out_a = k0 * cos_theta - k1 * sin_theta;
            uint q_dst_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            uint k_flat_b = i1 * ne00 + idx_b;
            uint k_dst_b = op9 + k_flat_b * op10 + (uint)row_idx * op11 + i3 * nb3;
            store_auto(dst, q_dst_b, (q0 * sin_theta + q1 * cos_theta) * q_scale, dst_esize);
            store_auto(temp, k_dst_b, k0 * sin_theta + k1 * cos_theta, op12);
        } else if (idx_b < ne00) {
            uint q_off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint k_off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, 0);
            uint q_dst_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            uint k_flat_b = i1 * ne00 + idx_b;
            uint k_dst_b = op9 + k_flat_b * op10 + (uint)row_idx * op11 + i3 * nb3;
            store_auto(dst, q_dst_b, asfloat(src0.Load(q_off_b)) * q_scale, dst_esize);
            store_auto(temp, k_dst_b, asfloat(src3.Load(k_off_b)), op12);
        }

        uint q_dst_a = offset_4d(idx_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        uint k_flat_a = i1 * ne00 + idx_a;
        uint k_dst_a = op9 + k_flat_a * op10 + (uint)row_idx * op11 + i3 * nb3;
        store_auto(dst, q_dst_a, q_out_a * q_scale, dst_esize);
        store_auto(temp, k_dst_a, k_out_a, op12);
    }
}
