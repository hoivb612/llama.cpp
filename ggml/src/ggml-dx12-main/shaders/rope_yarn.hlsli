// rope_yarn.hlsli - Shared YaRN scaling helpers for ROPE shaders
//
// Implements the YaRN extrapolation/interpolation mix and mscale damping
// from the paper "YaRN: Efficient Context Window Extension of Large Language
// Models" (https://arxiv.org/abs/2309.00071).
//
// When ext_factor == 0 the helper degenerates to standard ROPE
// (theta = freq_scale * theta_extrap, mscale unchanged) so callers can use
// it unconditionally without a separate fast path.

#ifndef ROPE_YARN_HLSLI
#define ROPE_YARN_HLSLI

static float rope_yarn_ramp(float low, float high, uint pair) {
    float y = ((float)pair - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void rope_yarn(float theta_extrap, float freq_scale, float corr_low, float corr_high,
                      uint pair, float ext_factor, float mscale,
                      out float cos_theta, out float sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    sincos(theta, sin_theta, cos_theta);
    cos_theta *= mscale;
    sin_theta *= mscale;
}

// Post-op ROPE for Q/K projection matvec fusion (see the slot map in
// ggml_common.hlsli). Rotates the (row0,row1) NORMAL-mode pair — sum0/sum1 are
// the two matvec dot products of a rotation pair — mirroring rope.hlsl /
// rope_set_rows.hlsl exactly (including partial-rotation passthrough and YaRN).
// pair_in_head is the rotation pair index within the head. Position rides
// src2.Load(0) (single-token M=1 decode); freq_factors ride src4.
void mmv_rope_pair(uint pair_in_head, float sum0, float sum1,
                   out float out0, out float out1) {
    uint  n_dims      = op1;
    float freq_base   = asfloat(op3);
    float freq_scale  = asfloat(op4);
    float ext_factor  = asfloat(op5);
    float attn_factor = asfloat(op6);
    float corr_low    = asfloat(op8);
    float corr_high   = asfloat(op9);
    uint  has_ff      = op11;

    if (pair_in_head >= n_dims / 2u) {
        // Partial-rotation passthrough: elements beyond n_dims copy unchanged.
        out0 = sum0;
        out1 = sum1;
        return;
    }

    int pos = asint(src2.Load(0));
    float theta_extrap = (float)pos * exp2(-(float)(pair_in_head * 2u) / (float)n_dims * log2(freq_base));
    if (has_ff != 0u) {
        float ff = asfloat(src4.Load(pair_in_head * 4u));
        theta_extrap = theta_extrap / ff;
    }

    float cos_theta, sin_theta;
    rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, pair_in_head, ext_factor, attn_factor, cos_theta, sin_theta);

    out0 = sum0 * cos_theta - sum1 * sin_theta;
    out1 = sum0 * sin_theta + sum1 * cos_theta;
}

#endif // ROPE_YARN_HLSLI
