// rope.hlsl - Rotary Positional Embedding
// Supports NORMAL (mode=0) and NEOX (mode=2) styles.
//
// NORMAL: pairs are (2i, 2i+1)       → pattern [cscscscs]
// NEOX:   pairs are (i, i+n_dims/2)  → pattern [ccccssss]
//
// op_params layout (matches ggml ROPE):
//   [0]=n_past(0), [1]=n_dims, [2]=mode, [3]=n_ctx(0), [4]=n_ctx_orig
//   [5]=freq_base(float), [6]=freq_scale(float), [7]=ext_factor(float)
//   [8]=attn_factor(float), [9]=beta_fast(float), [10]=beta_slow(float)
//   [11..14]=mrope sections (mrope only; unused here)
//   [15]=has_ff (set host-side: 1 if src2/freq_factors is bound)
//
// Supports:
//   - F16/F32/BF16 src0 + dst via load_auto/store_auto
//   - attn_factor scaling of cos/sin
//   - per-pair freq_factors from src2 (has_ff != 0)
//   - YaRN ext_factor scaling (corr_low/high computed in-shader from
//     n_ctx_orig at op_params[4], beta_fast at op_params[9], beta_slow at op_params[10])
#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

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
    float freq_base   = op_param_f32(5);
    float freq_scale  = op_param_f32(6);
    float ext_factor  = op_param_f32(7);
    float attn_factor = op_param_f32(8);
    float beta_fast   = op_param_f32(9);
    float beta_slow   = op_param_f32(10);
    uint  n_ctx_orig  = op_param_uint(4);
    uint  has_ff      = op_param_uint(15);

    bool is_neox = (mode & 2u) != 0;

    uint half_dims = n_dims / 2;

    // Passthrough: elements beyond n_dims are copied as-is (esize-aware)
    if (pair >= half_dims) {
        uint pass_idx = n_dims + 2 * (pair - half_dims);
        uint pass_a = pass_idx;
        uint pass_b = pass_idx + 1;
        if (pass_a < ne0) {
            uint off_a = offset_4d(pass_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint od_a  = offset_4d(pass_a, i1, i2, i3, nb0,  nb1,  nb2,  nb3,  dst_offset);
            float v = load_auto(src0, off_a, src0_esize);
            store_auto(dst, od_a, v, dst_esize);
        }
        if (pass_b < ne0) {
            uint off_b = offset_4d(pass_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            uint od_b  = offset_4d(pass_b, i1, i2, i3, nb0,  nb1,  nb2,  nb3,  dst_offset);
            float v = load_auto(src0, off_b, src0_esize);
            store_auto(dst, od_b, v, dst_esize);
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

    // theta_extrap = pos * freq_base^(-2*pair/n_dims)
    float theta_extrap = (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));

    // Per-pair frequency scaling (e.g. Llama-3.1, Phi-3 LongRope)
    if (has_ff != 0u) {
        float ff = asfloat(src2.Load(pair * 4));
        theta_extrap = theta_extrap / ff;
    }

    // YaRN correction range (host could precompute, but standalone path has
    // n_ctx_orig/beta_fast/beta_slow available so we compute in-shader).
    const float two_pi = 6.2831853071795864769f;
    float corr_start = floor((float)n_dims * log((float)n_ctx_orig / (beta_fast * two_pi)) / (2.0f * log(freq_base)));
    float corr_end   = ceil ((float)n_dims * log((float)n_ctx_orig / (beta_slow * two_pi)) / (2.0f * log(freq_base)));
    float corr_low   = max(0.0f, corr_start);
    float corr_high  = min((float)n_dims - 1.0f, corr_end);

    float cos_theta, sin_theta;
    rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, pair, ext_factor, attn_factor, cos_theta, sin_theta);

    uint off_a = offset_4d(idx_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    float x0 = load_auto(src0, off_a, src0_esize);
    float x1 = load_auto(src0, off_b, src0_esize);

    uint od_a = offset_4d(idx_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    uint od_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, od_a, x0 * cos_theta - x1 * sin_theta, dst_esize);
    store_auto(dst, od_b, x0 * sin_theta + x1 * cos_theta, dst_esize);
}
