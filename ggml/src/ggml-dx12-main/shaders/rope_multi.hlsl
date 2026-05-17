// rope_multi.hlsl - Multi-dimensional Rotary Positional Embedding (mrope)
//
// Used by Qwen3-VL and similar models with per-dimension position encoding.
// Each dimension pair uses a position from one of 4 planes (temporal, height,
// width, extra) based on section assignments.
//
// op_params layout (copied from tensor->op_params):
//   [0]=n_past(0), [1]=n_dims, [2]=mode, [3]=n_ctx(0), [4]=n_ctx_orig
//   [5]=freq_base(float), [6]=freq_scale(float), [7]=ext_factor(float)
//   [8]=attn_factor(float), [9]=beta_fast(float), [10]=beta_slow(float)
//   [11..14]=sections[4] (int32)
//
// src0: input tensor (F32)
// src1: position tensor (I32) — shape includes 4 position planes at ne02 offsets
// dst:  output tensor (F32)

#include "ggml_common.hlsli"

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

    uint n_dims    = op_param_uint(1);
    float freq_base  = op_param_f32(5);
    float freq_scale = op_param_f32(6);
    float ext_factor = op_param_f32(7);
    float attn_factor = op_param_f32(8);
    float beta_fast = op_param_f32(9);
    float beta_slow = op_param_f32(10);

    // mrope sections: how many dimension pairs use each position plane
    int sect0 = asint(op_param_uint(11));
    int sect1 = asint(op_param_uint(12));
    int sect2 = asint(op_param_uint(13));
    int sect3 = asint(op_param_uint(14));

    uint mode = op_param_uint(2);
    bool is_vision = (mode & 16u) != 0;  // GGML_ROPE_TYPE_VISION = 24 (bit 4)
    bool is_imrope = (mode & 32u) != 0;  // GGML_ROPE_TYPE_IMROPE = 40 (bit 5)

    // Vision ROPE uses n_dims as pair offset; mrope uses n_dims/2
    uint half_dims = is_vision ? n_dims : (n_dims / 2);

    // Passthrough: elements beyond active dims are copied unchanged
    if (pair >= half_dims) {
        uint pass_a = pair;
        uint pass_b = pair + half_dims;
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

    // Neox-style pair mapping: (pair, pair + offset)
    uint idx_a = pair;
    uint idx_b = pair + half_dims;

    // ne02 from src0 — used as stride between position planes in src1
    uint ne02_pos = ne02;

    // Determine position and theta based on mode
    int pos;
    float theta_extrap;
    uint theta_pair = pair;
    uint yarn_pair = pair;

    if (is_vision) {
        // ROPE_TYPE_VISION: 2 sections with section-relative frequency exponents
        int sect_dims = sect0 + sect1;
        int sector = (int)pair % sect_dims;

        if (sector < sect0) {
            pos = asint(src1.Load(src1_offset + i2 * 4));
            // Section-relative: theta = pos * pow(theta_scale, sector)
            theta_pair = (uint)sector;
            theta_extrap = (float)pos * exp2(-(float)(theta_pair * 2) / (float)n_dims * log2(freq_base));
        } else {
            uint p0 = (uint)(sector - sect0);
            pos = asint(src1.Load(src1_offset + (i2 + ne02_pos) * 4));
            theta_pair = p0;
            theta_extrap = (float)pos * exp2(-(float)(theta_pair * 2) / (float)n_dims * log2(freq_base));
        }
    } else {
        // ROPE_TYPE_MROPE / IMROPE: 4 sections with global frequency exponent
        int sect_dims = sect0 + sect1 + sect2 + sect3;
        int sector = (int)pair % sect_dims;

        if (is_imrope) {
            // Interleaved: dims cycle through t/h/w sections
            if (sector % 3 == 1 && sector < 3 * sect1) {
                pos = asint(src1.Load(src1_offset + (i2 + ne02_pos) * 4));
            } else if (sector % 3 == 2 && sector < 3 * sect2) {
                pos = asint(src1.Load(src1_offset + (i2 + ne02_pos * 2) * 4));
            } else if (sector % 3 == 0 && sector < 3 * sect0) {
                pos = asint(src1.Load(src1_offset + i2 * 4));
            } else {
                pos = asint(src1.Load(src1_offset + (i2 + ne02_pos * 3) * 4));
            }
        } else {
            // Contiguous: [sect0 | sect1 | sect2 | sect3]
            int sec_w = sect0 + sect1;
            if (sector < sect0) {
                pos = asint(src1.Load(src1_offset + i2 * 4));
            } else if (sector < sec_w) {
                pos = asint(src1.Load(src1_offset + (i2 + ne02_pos) * 4));
            } else if (sector < sec_w + sect2) {
                pos = asint(src1.Load(src1_offset + (i2 + ne02_pos * 2) * 4));
            } else {
                pos = asint(src1.Load(src1_offset + (i2 + ne02_pos * 3) * 4));
            }
        }

        theta_pair = pair;
        theta_extrap = (float)pos * exp2(-(float)(theta_pair * 2) / (float)n_dims * log2(freq_base));
    }

    const float two_pi = 6.2831853071795864769f;
    float corr_start = floor((float)n_dims * log((float)op_param_uint(4) / (beta_fast * two_pi)) / (2.0f * log(freq_base)));
    float corr_end   = ceil ((float)n_dims * log((float)op_param_uint(4) / (beta_slow * two_pi)) / (2.0f * log(freq_base)));
    float corr_low = max(0.0f, corr_start);
    float corr_high = min((float)n_dims - 1.0f, corr_end);

    float cos_theta, sin_theta;
    rope_yarn(theta_extrap, freq_scale, corr_low, corr_high, yarn_pair, ext_factor, attn_factor, cos_theta, sin_theta);

    uint off_a = offset_4d(idx_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    float x0 = asfloat(src0.Load(off_a));
    float x1 = asfloat(src0.Load(off_b));

    uint od_a = offset_4d(idx_a, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    uint od_b = offset_4d(idx_b, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    dst.Store(od_a, asuint(x0 * cos_theta - x1 * sin_theta));
    dst.Store(od_b, asuint(x0 * sin_theta + x1 * cos_theta));
}
