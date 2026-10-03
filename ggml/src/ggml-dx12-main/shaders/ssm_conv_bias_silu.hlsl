// ssm_conv_bias_silu.hlsl - SSM_CONV fused with a channel-wise bias ADD and
// a trailing SiLU activation.  Bias is bound at src2 as a contiguous F32
// vector with one element per output channel (i1).
// See ssm_conv.hlsl for the base op layout.
#define APPLY_BIAS 1
#define APPLY_SILU 1
#include "ssm_conv_impl.hlsli"
