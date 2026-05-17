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

#endif // ROPE_YARN_HLSLI
