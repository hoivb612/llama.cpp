// ssm_conv_impl.hlsli - Shared body for ssm_conv{,_silu,_bias_silu}.
//
// Defines (set by the including .hlsl):
//   APPLY_BIAS - if 1, read a per-channel bias from src2[i1] and add to sum
//   APPLY_SILU - if 1, apply x*sigmoid(x) before the final store
//
// Both default to 0; the plain SSM_CONV shader includes this header with
// neither defined and is bit-identical to the original implementation.
//
// See ssm_conv.hlsl for the inputs/outputs/strides.
#ifndef APPLY_BIAS
#define APPLY_BIAS 0
#endif
#ifndef APPLY_SILU
#define APPLY_SILU 0
#endif

#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2;
    if (idx >= total) return;

    uint i1 = idx % ne0;
    uint rem = idx / ne0;
    uint i2 = rem % ne1;
    uint i3 = rem / ne1;

    uint nc = ne10;

    uint s_base = src0_offset + i3 * nb02 + i1 * nb01 + i2 * nb00;
    uint c_base = src1_offset + i1 * nb11;

    float sum = 0.0f;
    for (uint i0 = 0; i0 < nc; ++i0) {
        float s_val = asfloat(src0.Load(s_base + i0 * nb00));
        float c_val = asfloat(src1.Load(c_base + i0 * nb10));
        sum += s_val * c_val;
    }

#if APPLY_BIAS
    // Bias is a contiguous F32 vector with one element per output channel (i1).
    // Bound at src2; baked-in tensor offset (matches the bias-fusion convention
    // used by the matvec path).
    sum += asfloat(src2.Load(i1 * 4u));
#endif

#if APPLY_SILU
    sum = sum / (1.0f + exp(-sum));
#endif

    uint d_off = dst_offset + i3 * nb2 + i2 * nb1 + i1 * nb0;
    dst.Store(d_off, asuint(sum));
}
