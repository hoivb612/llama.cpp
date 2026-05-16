// rope.hlsl - Rotary Positional Embedding
// Supports NORMAL (mode=0) and NEOX (mode=2) styles.
//
// NORMAL: pairs are (2i, 2i+1)       → pattern [cscscscs]
// NEOX:   pairs are (i, i+n_dims/2)  → pattern [ccccssss]
//
// op_params layout:
//   [0]=n_past(0), [1]=n_dims, [2]=mode, [3]=n_ctx(0), [4]=n_ctx_orig
//   [5]=freq_base(float), [6]=freq_scale(float), [7]=ext_factor(float)
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint n_pairs = ne0 / 2;
    uint total_pairs = n_pairs * ne1 * ne2 * ne3;
    if (idx >= total_pairs) return;

    uint i3 = idx / (n_pairs * ne1 * ne2);
    uint rem = idx % (n_pairs * ne1 * ne2);
    uint i2 = rem / (n_pairs * ne1);
    rem = rem % (n_pairs * ne1);
    uint i1 = rem / n_pairs;
    uint pair = rem % n_pairs;

    uint n_dims = op_param_uint(1);
    uint mode   = op_param_uint(2);
    float freq_base  = op_param_f32(5);
    float freq_scale = op_param_f32(6);

    bool is_neox = (mode & 2u) != 0;

    uint half_dims = n_dims / 2;

    // Passthrough: elements beyond n_dims are copied
    if (pair >= half_dims) {
        uint pass_idx = n_dims + 2 * (pair - half_dims);
        uint pass_a = pass_idx;
        uint pass_b = pass_idx + 1;
        if (pass_a < ne0) {
            uint off_a = offset_4d(pass_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint od_a  = offset_4d(pass_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            dst.Store(od_a, src0.Load(off_a));
        }
        if (pass_b < ne0) {
            uint off_b = offset_4d(pass_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint od_b  = offset_4d(pass_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            dst.Store(od_b, src0.Load(off_b));
        }
        return;
    }

    // Rotation pair: compute element indices
    uint idx_a, idx_b;
    if (is_neox) {
        idx_a = pair;
        idx_b = pair + half_dims;
    } else {
        idx_a = pair * 2;
        idx_b = pair * 2 + 1;
    }

    // Position from src1 (int32)
    uint pos_off = src1_offset + i2 * nb10;
    int pos = asint(src1.Load(pos_off));

    // Optimized theta: avoid pow() by using exp2(x * log2(base))
    // pow(freq_base, -2*pair/n_dims) = exp2(-2*pair/n_dims * log2(freq_base))
    float theta = (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));
    theta *= freq_scale;

    float cos_theta, sin_theta;
    sincos(theta, sin_theta, cos_theta);

    uint off_a = offset_4d(idx_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    float x0 = asfloat(src0.Load(off_a));
    float x1 = asfloat(src0.Load(off_b));

    uint od_a = offset_4d(idx_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    uint od_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    dst.Store(od_a, asuint(x0 * cos_theta - x1 * sin_theta));
    dst.Store(od_b, asuint(x0 * sin_theta + x1 * cos_theta));
}
